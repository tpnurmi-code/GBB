import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np


from nilearn.glm.first_level import compute_regressor
from nilearn.image import resample_to_img, coord_transform
from nilearn.maskers import NiftiLabelsMasker
import nilearn.datasets
import nilearn.image
from nilearn.glm.first_level import glover_hrf
import nilearn.plotting as plotting
from nilearn import datasets


from scipy.ndimage import center_of_mass
from scipy.spatial.distance import cdist
from scipy.stats import zscore
import scipy.io
import scipy.signal
from scipy.stats import gamma


import config
import os
import json

from tqdm import tqdm  # <--- This fixes it



def canonical_hrf_response(time):
# Time axis for HRF (32s is sufficient)
        # We ensure at least one point exists to avoid sum=0
     hrf_time = np.arange(0, time, config.TR)
     if len(hrf_time) == 0: hrf_time = np.array([0.0])

        # Standard Glover/SPM parameters: Peak at 6s
        # Gamma PDF: (t, a, scale) -> Peak is at (a-1)*scale
        # For peak at 6s with scale 1: a=7.
     hrf_kernel = gamma.pdf(hrf_time, 7, scale=1)
        
     # Normalize (Safety check to prevent NaN)
     if np.sum(hrf_kernel) > 1e-9:
         hrf_kernel /= np.sum(hrf_kernel)
     else:
         # Fallback for weird TRs: Unit Impulse
         hrf_kernel = np.zeros_like(hrf_time)
         hrf_kernel[0] = 1.0
         print(f"⚠️ Warning: HRF sum was 0. Using Impulse fallback for TR={config.TR}")
     return hrf_kernel

def canonical_cbv_response(tr, time_length=32.0, onset=0.0):
    """
    Generates a canonical Cerebral Blood Volume (CBV) response function.
    
    Compared to BOLD (Glover):
    - Peaks slightly earlier (~4s vs 5-6s).
    - Has a narrower dispersion.
    - Lacks the pronounced undershoot (dip).
    
    Parameters:
    - tr: Repetition time in seconds.
    - time_length: Duration of the kernel in seconds.
    - onset: Delay before onset.
    
    Returns:
    - cbv_kernel: Normalized array of the CBV response.
    """
    dt = tr # Temporal resolution
    time_axis = np.arange(0, time_length, dt) - onset
    
    # Parameters for CBV (approximated from Buxton/Mandeville Balloon Model)
    # Peak at ~4s, Width ~4s
    # Using scipy Gamma PDF: gamma.pdf(x, a, loc, scale)
    # Mean = a * scale. We want Mean ~ 4s. Let scale=1, a=4.
    
    # Shape parameter (alpha)
    a = 4.0 
    # Scale parameter (theta)
    scale = 1.0 
    
    # Generate response
    cbv_response = gamma.pdf(time_axis, a, scale=scale)
    
    # Zero out negative time (pre-onset)
    cbv_response[time_axis < 0] = 0
    
    # Normalize to max 1 (Unit Height)
    if np.max(cbv_response) > 0:
        cbv_response /= np.max(cbv_response)
        
    return cbv_response

class NiftiLaminarDataset(Dataset):
 
    def __init__(self, data_list, mask_img, window_size=30, run_type='train', sensory_regions=None):
        self.window_size = window_size
        self.tr = config.TR
        self.prediction_horizon = config.PREDICTION_HORIZON  # also fixes later AttributeError
    
        # Ensure we have a NIfTI image object for the parcellation/labels
        if isinstance(mask_img, (str, os.PathLike)):
            self.parcellation_img = nib.load(mask_img)
        else:
            self.parcellation_img = mask_img
        
        if run_type=='train':
            self.is_training = True
        else:
             self.is_training = False
        # --- PART 1: SETUP MASKER & METADATA ---
        
        # 1. Initialize Masker
        self.masker = NiftiLabelsMasker(
            labels_img=self.parcellation_img,
            standardize=False,
            detrend=config.DETREND,
            t_r=self.tr,  # FIX: self.TR doesn't exist
            low_pass=config.LOW_PASS,
            high_pass=config.HIGH_PASS
        )
        self.masker.fit()
        
        # 2. Get Number of Nodes & Mask Object

        
        mask_img_obj = nib.load(mask_img)
        mask_data = mask_img_obj.get_fdata()
        affine = mask_img_obj.affine
        
        found_ids = np.unique(mask_data)
        found_ids = found_ids[found_ids != 0]
        found_ids.sort()
        self.num_nodes = len(found_ids)
        
        # 3. Load Metadata / Calculate Centroids (THE FIX)
        self.sensory_indices = []
        self.region_labels = []
        
        metadata_path = os.path.join(os.path.dirname(mask_img), "mask_metadata.json")
        
        coords = []
        labels = []
        
        # A. Try Loading JSON
        if os.path.exists(metadata_path):
            print(f"  -> Loading metadata from {metadata_path}...")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract in sorted order
            for node_id in sorted([int(k) for k in metadata.keys()]):
                info = metadata[str(node_id)]
                if 'centroid_mni' in info:
                    coords.append(info['centroid_mni'])
                else:
                    # Partial fallback if JSON exists but lacks coords
                    voxel_center = center_of_mass(mask_data == node_id)
                    mni_center = nib.affines.apply_affine(affine, voxel_center)
                    coords.append(mni_center)
                
                # Use real label if available
                labels.append(info.get('label', f"Region_{node_id}"))
            
            coords = np.array(coords)

        # B. Fallback (If JSON missing or empty)
        if len(coords) == 0:
            print(f"  -> Computing centroids for {self.num_nodes} regions (Fallback)...")
            
            for node_id in found_ids:
                # Center of Mass (Voxels) -> MNI (Millimeters)
                voxel_center = center_of_mass(mask_data == node_id)
                mni_center = nib.affines.apply_affine(affine, voxel_center)
                coords.append(mni_center)
                
                # Generate Generic Label (Silences Warnings)
                labels.append(f"Region_{int(node_id)}")
                
            coords = np.array(coords)

        # [CRITICAL] Save to Class Attributes
        self.coords = coords 
        self.region_labels = labels
        
        # Debug Physics
        if len(self.coords) > 0:
            target_debug = np.array(config.STIMULUS_MNI_COORDS, dtype=float)
            dists = np.linalg.norm(self.coords - target_debug, axis=1)
            print(f"     [PHYSICS CHECK] Closest node to Hand Knob: {np.min(dists):.2f}mm")
            
            if len(self.sensory_indices) > 0:
                sx = float(np.median(self.coords[self.sensory_indices, 0]))
                tx = float(target_debug[0])
                if np.sign(sx) != 0 and np.sign(tx) != 0 and np.sign(sx) != np.sign(tx):
                    print(f"⚠️ Sensory sphere hemisphere mismatch? sensory median x={sx:.1f}, target x={tx:.1f}")


        # --- PART 2: STIMULUS INJECTION LOGIC ---
        print(f"  -> Stimulus Mode: {config.STIMULUS_INJECTION_MODE}")
        
        if config.STIMULUS_INJECTION_MODE == "REGION_NAME":
            # Legacy Mode
            if sensory_regions is None: sensory_regions = config.SENSORY_REGIONS
            self.sensory_indices = [
                i for i, lbl in enumerate(self.region_labels)
                if any(x in lbl for x in sensory_regions)
            ]
            
        elif config.STIMULUS_INJECTION_MODE == "COORDINATES":
            # Physics Mode
            target_loc = np.array(config.STIMULUS_MNI_COORDS)
            dists = np.linalg.norm(self.coords - target_loc, axis=1)
            self.sensory_indices = np.where(dists < config.STIMULUS_RADIUS_MM)[0].tolist()
            print(f"     Target: {target_loc}, Radius: {config.STIMULUS_RADIUS_MM}mm")
        
        elif config.STIMULUS_INJECTION_MODE == "COORDS_REGION_INTERSECTION":
            # ---------------------------------------------------------
            # 1. PHYSICS: Get Candidate Nodes (The Sphere)
            # ---------------------------------------------------------
            target_loc = np.array(config.STIMULUS_MNI_COORDS)
            dists = np.linalg.norm(self.coords - target_loc, axis=1)
            sphere_indices = np.where(dists < config.STIMULUS_RADIUS_MM)[0]
            
            print(f"  [Physics] Found {len(sphere_indices)} nodes within {config.STIMULUS_RADIUS_MM}mm.")
            
            # DEBUG: Print the first few labels found in the sphere to verify naming convention
            if len(sphere_indices) > 0:
                sample_labels = [self.region_labels[i] for i in sphere_indices[:5]]
                print(f"  [Debug] Sample labels in sphere: {sample_labels}")

            # ---------------------------------------------------------
            # 2. ANATOMY: Broad String Matching (The Fix)
            # ---------------------------------------------------------
            anatomical_indices = []
            
            # Define broad sensory terms (Schaefer/AAL compatible)
            # We look for ANY of these substrings
            sensory_keywords = config.SENSORY_REGIONS
            # We explicitly exclude these to avoid Motor/Frontal bleed
            motor_keywords = config.EXCLUDED_REGIONS

            for i in sphere_indices:
                label = self.region_labels[i]
                
                # Check 1: Must act like Sensory
                is_sensory = any(k in label for k in sensory_keywords)
                # Check 2: Must NOT be Motor (Crucial for Sulcus separation)
                is_motor = any(k in label for k in motor_keywords)
                
                if is_sensory and not is_motor:
                    anatomical_indices.append(i)
                    # print(f"    -> Keeping Sensory Node: {label}") # Uncomment for verbose

            # ---------------------------------------------------------
            # 3. INTERSECTION & FALLBACK
            # ---------------------------------------------------------
            # We don't need set intersection anymore because we iterated over sphere_indices directly
            self.sensory_indices = anatomical_indices

            # CRITICAL FALLBACK: If string matching fails entirely (e.g., weird atlas labels)
            if len(self.sensory_indices) == 0:
                print(f"  ⚠️ WARNING: Anatomy filter removed all nodes! Switching to GEOMETRIC FALLBACK.")
                print(f"  -> Selecting nodes POSTERIOR to the Hand Knob (Y < {target_loc[1]}).")
                
                # Geometric Logic: Hand Knob Y is usually ~ -25. 
                # Anything more negative (e.g. -30, -40) is Posterior (Sensory).
                # Anything more positive (e.g. -10, 0) is Anterior (Motor).
                knob_y_coordinate = target_loc[1]
                
                fallback_indices = []
                for i in sphere_indices:
                    node_y = self.coords[i, 1]
                    if node_y < knob_y_coordinate:
                        fallback_indices.append(i)
                
                self.sensory_indices = fallback_indices
                print(f"  -> Fallback recovered {len(self.sensory_indices)} posterior nodes.")

            print(f"  [Final] Stimulus injecting into {len(self.sensory_indices)} nodes.")
            
        # Safety Fallback
        if len(self.sensory_indices) == 0:
            print("⚠️ WARNING: No sensory nodes matched. Defaulting to ALL OPEN.")
            self.sensory_indices = list(range(self.num_nodes))

        # [FIX] Convert to Tensor for PyTorch Model
        mask_np = np.zeros(self.num_nodes, dtype=np.float32)
        mask_np[self.sensory_indices] = 1.0
        self.sensory_mask = torch.tensor(mask_np, dtype=torch.float32)


        # --- PART 3: LOAD DATA (RESTORED ORIGINAL FEATURES) ---
        concat_series = []
        concat_stim = []
        
        # Use tqdm for progress bar
        iterator = data_list
        if len(data_list) > 1:
            iterator = tqdm(data_list, desc="Processing runs", leave=False)
        
        run_lengths = []
        
        for item in iterator:
            # A. Load & Resample fMRI (Explicit Resampling Restored)
            img = nib.load(item['fmri'])
            
            # Resample to match mask geometry (Handles grid mismatches)
            resampled_func = resample_to_img(
                source_img=img,
                target_img=mask_img_obj,
                interpolation='continuous'
            )
            
            # Extract Signals
            fmri_data = self.masker.transform(resampled_func)
            
            # B. Load Stimulus (Restored Path Logic)
            run_len = fmri_data.shape[0]
            
            if config.STIMULUS_MODE == "EVENTS":
                stim_drive = self._load_events_stimulus(item['events'], run_len)
            elif config.STIMULUS_MODE == "DENSE":
                # Check events folder first, then fmri folder
                mat_path = item['events'].replace("_events.tsv", config.DENSE_STIMULUS_EXT)
                if not os.path.exists(mat_path):
                    mat_path = item['fmri'].replace(".nii", config.DENSE_STIMULUS_EXT)
                stim_drive = self._load_dense_stimulus(mat_path, run_len)
            else:
                stim_drive = np.zeros((run_len, 1))

            # C. Post-Processing (Z-Score)
            fmri_data = zscore(fmri_data, axis=0)
            fmri_data = np.nan_to_num(fmri_data)
            
            concat_series.append(fmri_data)
            concat_stim.append(stim_drive)
            run_lengths.append(run_len)
            
        # Concatenate all runs
        self.time_series = np.concatenate(concat_series, axis=0).astype(np.float32)
        self.stim_drive = np.concatenate(concat_stim, axis=0).astype(np.float32)
        
        self.run_lengths = run_lengths  # store per-run lengths (populate this list in the loop)
        self.run_offsets = np.cumsum([0] + self.run_lengths[:-1]).tolist()
        
        # Precompute valid window start indices that do NOT cross run boundaries
        max_start_per_run = [
            L - (self.window_size + self.prediction_horizon) for L in self.run_lengths
        ]
        self.valid_start_indices = []
        for off, max_s in zip(self.run_offsets, max_start_per_run):
            if max_s >= 0:
                self.valid_start_indices.extend(list(range(off, off + max_s + 1)))
        
        self.total_time = self.time_series.shape[0]
        
        
        
        # --- PART 4: ADJACENCY & GROUPING (RESTORED) ---
        print("  -> Computing Spatial Adjacency Matrix...")
        
        # 1. Calculate Distance Matrix (using self.coords)
        dist_matrix = cdist(self.coords, self.coords, metric='euclidean')
        
        self.distance_matrix = torch.tensor(dist_matrix, dtype=torch.float32)
        self.node_coords = plotting.find_parcellation_cut_coords(mask_img)
        
        # ======================================================
        # [NEW] CORTICAL COLUMN / LAMINAR SHIELD LOGIC
        # ======================================================
        if hasattr(config, 'COLUMNAR_MASK_FILE') and os.path.exists(config.COLUMNAR_MASK_FILE):
            print(f"🛡️ 7T Shield Detected: Loading Columnar Mask from {config.COLUMNAR_MASK_FILE}")
            col_img = nib.load(config.COLUMNAR_MASK_FILE)
            col_img = resample_to_img(col_img, mask_img, interpolation='nearest')
            
            
            # Extract the column ID for each of the 305 nodes
            col_data = col_img.get_fdata()
            col_ids = []
            
            for node_id in found_ids:
                vals = col_data[mask_data == node_id]
                vals = vals[vals > 0].astype(int)
                if len(vals) == 0:
                    col_ids.append(0)
                else:
                    col_ids.append(np.bincount(vals).argmax())
            
            self.column_ids = torch.tensor(col_ids, dtype=torch.long)
        else:
            # 3T MODE: SCHAEFER 2018 (100 Parcels)
            # This is the "Best of Both Worlds":
            # 1. Data-Driven: Derived from fMRI clustering (Yeo Lab).
            # 2. Robust: Groups ~3 nodes together to kill salt-and-pepper noise.
            # 3. Precise: Splits large gyri (like Precentral) into functional sub-units.
            
            print("🌐 3T Mode: Using Schaefer 2018 (100 Parcellation) for Data-Driven Grouping...")
            
            # Fetch the atlas (MNI152, 100 parcels, 7 networks resolution)
            schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
            atlas_filename = schaefer.maps
            
            atlas_img = nib.load(atlas_filename)
            atlas_data = atlas_img.get_fdata()
            affine = atlas_img.affine
            
            labels = []
            for (x, y, z) in self.node_coords:
                # Transform MNI (mm) -> Voxel Indices
                # inverse affine * [x,y,z,1]
                inv_affine = np.linalg.inv(affine)
                vox_idx = np.dot(inv_affine, [x, y, z, 1])[:3]
                i, j, k = np.round(vox_idx).astype(int)
                
                # Extract label
                try:
                    label = atlas_data[i, j, k]
                except IndexError:
                    label = 0 # Background
                
                labels.append(label)
            
            # Save as Column IDs
            self.column_ids = torch.tensor(labels, dtype=torch.float32)
            
            # Quick Stats for the Log
            unique_labels = len(np.unique(labels))
            print(f"   -> Mapped 305 Nodes to {unique_labels} Functional Schaefer Parcels.")
            print("   -> Group Lasso will now enforce consistency within these functional sub-regions.")
        
        
        # 2. Convert to Adjacency (Gaussian Kernel)
        sigma = config.SMOOTHNESS_SIGMA_MM if hasattr(config, 'SMOOTHNESS_SIGMA_MM') else 10.0
        adj = np.exp(- (dist_matrix**2) / (2 * sigma**2))
        adj[adj < 0.01] = 0 # Threshold
        
        self.adjacency = torch.tensor(adj, dtype=torch.float32)
        print(f"  -> Adjacency Matrix ready: {self.adjacency.shape}")
        
        # 3. Group by Anatomy (Restored)
        unique_region_names = sorted(list(set(self.region_labels)))
        name_to_int = {name: i for i, name in enumerate(unique_region_names)}
        node_group_ids = [name_to_int[label] for label in self.region_labels]
        self.node_region_ids = torch.tensor(node_group_ids, dtype=torch.long)
        
        print(f"  -> Physics Constraint: 'Laminar Shield' configured for {len(self.node_region_ids)} nodes.")       
        
    def _recover_aal_labels(self, found_ids):
        """Maps integer IDs (e.g. 2001) back to AAL string names."""
        import nilearn.datasets
        aal = nilearn.datasets.fetch_atlas_aal()
        
        # Build Lookup Table: Int ID -> String Name
        # AAL indices are often strings ('2001'), so we convert to int for safety
        id_to_name = {}
        for idx, label in zip(aal.indices, aal.labels):
            try:
                id_to_name[int(idx)] = str(label)
            except ValueError:
                continue
            
        restored_labels = []
        for roi_id in found_ids:
            # roi_id comes from numpy as float usually (2001.0), convert to int
            val = int(roi_id)
            
            if val in id_to_name:
                restored_labels.append(id_to_name[val])
            else:
                # Fallback if AAL doesn't match
                print(f"WARNING: Region ID {val} not found in AAL atlas.")
                restored_labels.append(f"Region_{val}")
                
        return restored_labels


    def _make_sensory_mask(self, loop):
        # Find the integer IDs for these regions
        target_indices = []
        dataset = nilearn.datasets.fetch_atlas_aal()
        mask_tensor = torch.zeros(self.num_nodes, 1)
        
        labels = dataset.labels
                
        found_regions = 0
        #print("Selected Regions:")
        # Iterate over the labels that match the data rows
        for i, label in enumerate(self.region_labels):
            # Convert label to string just in case
            label_str = str(label)
            
            # Check for config keywords (e.g. "Postcentral", "Visual")
            if any(k in label_str for k in config.SENSORY_REGIONS):
                # i is the ROW INDEX. This is correct.
                mask_tensor[i] = 1.0 
                found_regions += 1
                # print(f"  + Open Sensory Port: {label_str} (Row {i})")
        
        if found_regions == 0:
            print("WARNING: No sensory regions found! Defaulting to ALL OPEN.")
            mask_tensor[:] = 1.0
        else:
            loop.set_postfix_str(f"  -> Opened {found_regions} / {self.num_nodes} sensory ports.")
            
        return mask_tensor
            
    def _load_events_stimulus(self, events_path, n_scans):
        """ Classic SPM-style Event convolution """
        import pandas as pd
        events = pd.read_csv(events_path, sep='\t')
        events = events[events['trial_type'] == 'hand_movement'] # Filter if needed
        
        frame_times = np.arange(n_scans) * self.tr
        events_for_nilearn = events[['onset', 'duration', 'trial_type']]
        
        stim_reg, _ = compute_regressor(
            events_for_nilearn, 
            "spm", 
            frame_times 
        )
        # Shape (Time, 1)
        return zscore(stim_reg.squeeze()).astype(np.float32).reshape(-1, 1)

    def _load_dense_stimulus(self, file_path, target_n_scans):
        """
        Universal Continuous Data Loader (.mat / .npy)
    
        Handles 1D accelerometer-like signals, and can also tolerate
        higher-dimensional dense stimuli by flattening non-time dimensions
        after TR binning.
    
        Processing steps:
        1. Load dense stimulus file
        2. Standardize shape to (raw_time, channels...)
        3. Rectify 1D movement-energy signals
        4. Downsample/integrate raw samples into TR bins
        5. Optionally pre-convolve with HRF/CBV response
           - but do NOT pre-convolve by default if the model already has
             a hemodynamic observation head
        6. Pad/crop to fMRI length
        7. Z-score
        """
    
        # ---------------------------------------------------------
        # A. LOAD FILE
        # ---------------------------------------------------------
        if file_path.endswith(".mat"):
            mat = scipy.io.loadmat(file_path)
    
            # Find the variable that is not MATLAB metadata.
            keys = [k for k in mat.keys() if not k.startswith("__")]
            if len(keys) == 0:
                raise ValueError(f"No usable variables found in dense stimulus file: {file_path}")
    
            raw_data = mat[keys[0]]
    
        elif file_path.endswith(".npy"):
            raw_data = np.load(file_path)
    
        else:
            raise ValueError(f"Unsupported dense file format: {file_path}")
    
        raw_data = np.asarray(raw_data, dtype=np.float32)
    
        # ---------------------------------------------------------
        # B. STANDARDIZE SHAPE: (Time, Channels...)
        # ---------------------------------------------------------
        # Common MATLAB case: (1, N) should become (N, 1)
        if raw_data.ndim == 2 and raw_data.shape[0] == 1 and raw_data.shape[1] > 1:
            raw_data = raw_data.T
    
        # If accidentally loaded as scalar / 1D, make it (Time, 1)
        if raw_data.ndim == 1:
            raw_data = raw_data[:, None]
    
        # ---------------------------------------------------------
        # C. RECTIFICATION: movement energy for 1D sensors
        # ---------------------------------------------------------
        if config.STIMULUS_INPUT_CHANNELS == 1:
            raw_data = np.abs(raw_data)
    
        # ---------------------------------------------------------
        # D. DOWNSAMPLE / INTEGRATE INTO TR BINS
        # ---------------------------------------------------------
        samples_per_tr = int(config.RAW_SAMPLING_RATE * self.tr)
        if samples_per_tr <= 0:
            raise ValueError(
                f"Invalid samples_per_tr={samples_per_tr}. "
                f"Check RAW_SAMPLING_RATE={config.RAW_SAMPLING_RATE} and TR={self.tr}."
            )
    
        n_available_trs = raw_data.shape[0] // samples_per_tr
    
        if n_available_trs <= 0:
            print(
                f"⚠️ Dense stimulus shorter than one TR: {file_path}. "
                f"Returning zero stimulus with {target_n_scans} scans."
            )
            return np.zeros((target_n_scans, config.STIMULUS_INPUT_CHANNELS), dtype=np.float32)
    
        cutoff = n_available_trs * samples_per_tr
        truncated_data = raw_data[:cutoff]
    
        # Shape: (n_trs, samples_per_tr, channels...)
        new_shape = (n_available_trs, samples_per_tr) + raw_data.shape[1:]
        binned_data = truncated_data.reshape(new_shape).mean(axis=1)
    
        # Flatten non-time dimensions into channels for the model.
        # For normal 1D stimuli this remains (T, 1).
        binned_data = binned_data.reshape(n_available_trs, -1).astype(np.float32)
    
        # ---------------------------------------------------------
        # E. DECIDE WHETHER TO PRE-CONVOLVE STIMULUS
        # ---------------------------------------------------------
        should_convolve_stim = bool(config.CONVOLVE_STIMULUS)
    
        # If the model already has a hemodynamic observation head, default to
        # raw/neural stimulus drive. This prevents double-HRF/double-hemodynamics.
        if getattr(config, "USE_HEMODYNAMIC_HEAD", False):
            should_convolve_stim = should_convolve_stim and getattr(
                config,
                "ALLOW_STIMULUS_PRECONV_WITH_HEMO",
                False
            )
    
        # ---------------------------------------------------------
        # F. OPTIONAL HRF/CBV PRE-CONVOLUTION
        # ---------------------------------------------------------
        if should_convolve_stim:
            if config.RESPONSE_FUNCTION == "cbv":
                canonical_response = canonical_cbv_response(config.TR, 30)
            else:
                canonical_response = canonical_hrf_response(time=30)
    
            canonical_response = np.asarray(canonical_response, dtype=np.float32)
    
            n_obs, n_ch = binned_data.shape
            convolved_data = np.zeros_like(binned_data, dtype=np.float32)
    
            for i in range(n_ch):
                conv = scipy.signal.convolve(
                    binned_data[:, i],
                    canonical_response,
                    mode="full"
                )
                convolved_data[:, i] = conv[:n_obs]
    
            convolved_data = np.nan_to_num(convolved_data).astype(np.float32)
    
            # Pad or crop to match fMRI length
            if convolved_data.shape[0] < target_n_scans:
                pad_len = target_n_scans - convolved_data.shape[0]
                convolved_data = np.pad(
                    convolved_data,
                    ((0, pad_len), (0, 0)),
                    mode="constant"
                )
            elif convolved_data.shape[0] > target_n_scans:
                convolved_data = convolved_data[:target_n_scans]
    
            processed_stim = (
                convolved_data - convolved_data.mean()
            ) / (convolved_data.std() + 1e-6)
    
        # ---------------------------------------------------------
        # G. RAW / NON-CONVOLVED STIMULUS PATH
        # ---------------------------------------------------------
        else:
            if binned_data.shape[0] < target_n_scans:
                pad_len = target_n_scans - binned_data.shape[0]
                padding = np.zeros((pad_len, binned_data.shape[1]), dtype=np.float32)
                binned_data = np.concatenate([binned_data, padding], axis=0)
    
            elif binned_data.shape[0] > target_n_scans:
                binned_data = binned_data[:target_n_scans]
    
            processed_stim = (
                binned_data - binned_data.mean()
            ) / (binned_data.std() + 1e-6)
    
        processed_stim = np.nan_to_num(processed_stim).astype(np.float32)
    
        return processed_stim



    def __len__(self):
        return len(self.valid_start_indices)

    def __getitem__(self, idx):
        start = self.valid_start_indices[idx]
        x_fmri = self.time_series[start: start + self.window_size]
        x_stim = self.stim_drive[start: start + self.window_size]
        y_target = self.time_series[
            start + self.window_size :
            start + self.window_size + config.PREDICTION_HORIZON
        ]  # shape: (Horizon, Nodes)
        x_stim_t = torch.tensor(x_stim, dtype=torch.float32)
        # Ensure shape is (T, C) not (T, C, 1)
        if x_stim_t.ndim == 1:
            x_stim_t = x_stim_t.unsqueeze(-1)
        
        return (
            torch.tensor(x_fmri, dtype=torch.float32),
            x_stim_t,
            #self.adjacency.clone().detach().float(),
            torch.tensor(y_target, dtype=torch.float32),
        )