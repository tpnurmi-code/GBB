def _select_sensory_indices(
        self,
        sensory_regions: list[str],
    ) -> list[int]:
        mode = str(config.STIMULUS_INJECTION_MODE).upper()

        labels_lower = [label.lower() for label in self.region_labels]
        sensory_terms = [term.lower() for term in sensory_regions]
        excluded_terms = [
            term.lower() for term in config.EXCLUDED_REGIONS
        ]

        name_indices = [
            index
            for index, label in enumerate(labels_lower)
            if any(term in label for term in sensory_terms)
            and not any(term in label for term in excluded_terms)
        ]

        target = np.asarray(
            config.STIMULUS_MNI_COORDS,
            dtype=float,
        )
        distances = np.linalg.norm(self.coords - target, axis=1)
        sphere_indices = np.where(
            distances <= float(config.STIMULUS_RADIUS_MM)
        )[0].tolist()

        if mode == "REGION_NAME":
            selected = name_indices
            fallback = sphere_indices

        elif mode == "COORDINATES":
            selected = sphere_indices
            fallback = name_indices

        elif mode == "COORDS_REGION_INTERSECTION":
            name_set = set(name_indices)
            selected = [
                index
                for index in sphere_indices
                if index in name_set
            ]

            posterior_sphere = [
                index
                for index in sphere_indices
                if self.coords[index, 1] < target[1]
            ]
            fallback = (
                posterior_sphere
                or sphere_indices
                or name_indices
            )

        else:
            raise ValueError(
                f"Unknown STIMULUS_INJECTION_MODE: {mode}"
            )

        if not selected:
            details = (
                "No sensory nodes matched the configured targeting rule. "
                f"mode={mode}, "
                f"sensory_regions={sensory_regions}, "
                f"name_matches={len(name_indices)}, "
                f"coordinate_matches={len(sphere_indices)}, "
                f"target={target.tolist()}, "
                f"radius_mm={float(config.STIMULUS_RADIUS_MM)}"
            )

            if self.sensory_selection_policy == "STRICT":
                raise ValueError(details)

            if self.sensory_selection_policy == "WARN":
                warnings.warn(
                    f"{details} Using the explicit fallback selector.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            selected = fallback

        if not selected and self.allow_all_nodes_stimulus:
            selected = list(range(self.num_nodes))

        if not selected:
            raise ValueError(
                "No sensory nodes were selected, including by the "
                "configured fallback. Set "
                "ALLOW_ALL_NODES_STIMULUS=True only for an intentional "
                "whole-network ablation."
            )

        selected = sorted(
            set(int(index) for index in selected)
        )

        if (
            len(selected) == self.num_nodes
            and not self.allow_all_nodes_stimulus
        ):
            raise ValueError(
                "Sensory selection resolved to every model node. Set "
                "ALLOW_ALL_NODES_STIMULUS=True only for an intentional "
                "whole-network ablation."
            )

        return selected

def make_sensory_mask(
    num_nodes: int,
    sensory_indices: list[int],
) -> torch.Tensor:
    ...
