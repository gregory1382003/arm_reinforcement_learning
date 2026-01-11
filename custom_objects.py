from robosuite.models.objects import CompositeObject


class TubeObject(CompositeObject):
    def __init__(
        self,
        name,
        radius,
        half_length,
        rgba=None,
        cap_radius=None,
        cap_half_length=None,
        cap_rgba=None,
        include_cap=True,
        joints="default",
    ):
        geom_types = ["cylinder"]
        geom_sizes = [[radius, half_length]]
        geom_locations = [[0.0, 0.0, 0.0]]

        use_rgba = rgba is not None or cap_rgba is not None
        geom_rgbas = None
        if use_rgba:
            base_rgba = rgba if rgba is not None else [0.8, 0.6, 0.6, 1.0]
            geom_rgbas = [base_rgba]

        total_half_size = [radius, radius, half_length]

        if include_cap:
            cap_radius = cap_radius if cap_radius is not None else radius * 1.2
            cap_half_length = cap_half_length if cap_half_length is not None else radius * 0.5
            geom_types.append("cylinder")
            geom_sizes.append([cap_radius, cap_half_length])
            geom_locations.append([0.0, 0.0, half_length + cap_half_length])
            if geom_rgbas is not None:
                cap_rgba = cap_rgba if cap_rgba is not None else geom_rgbas[0]
                geom_rgbas.append(cap_rgba)
            total_half_size = [
                max(radius, cap_radius),
                max(radius, cap_radius),
                half_length + cap_half_length,
            ]

        super().__init__(
            name=name,
            total_size=total_half_size,
            geom_types=geom_types,
            geom_sizes=geom_sizes,
            geom_locations=geom_locations,
            geom_rgbas=geom_rgbas,
            joints=joints,
            locations_relative_to_center=True,
        )


class PalletObject(CompositeObject):
    def __init__(
        self,
        name,
        base_half_size,
        slot_positions,
        base_rgba=None,
        slot_rgba=None,
        slot_size=0.006,
        hole_radius=None,
        hole_depth=None,
        hole_rgba=None,
        prototype_radius=None,
        prototype_half_length=None,
        prototype_rgba=None,
        prototype_center_z=None,
    ):
        geom_types = ["box"]
        geom_sizes = [list(base_half_size)]
        geom_locations = [[0.0, 0.0, 0.0]]

        use_rgba = base_rgba is not None or hole_rgba is not None
        geom_rgbas = None
        if use_rgba:
            base_rgba = base_rgba if base_rgba is not None else [0.6, 0.6, 0.6, 1.0]
            geom_rgbas = [base_rgba]

        if hole_radius is not None and hole_depth is not None:
            hole_rgba = hole_rgba if hole_rgba is not None else [0.08, 0.08, 0.08, 1.0]
            hole_half_depth = hole_depth / 2.0
            hole_center_z = base_half_size[2] - hole_half_depth
            for pos in slot_positions:
                geom_types.append("cylinder")
                geom_sizes.append([hole_radius, hole_half_depth])
                geom_locations.append([pos[0], pos[1], hole_center_z])
                if geom_rgbas is not None:
                    geom_rgbas.append(hole_rgba)
        if prototype_radius is not None and prototype_half_length is not None and prototype_center_z is not None:
            prototype_rgba = prototype_rgba if prototype_rgba is not None else [0.2, 0.2, 0.2, 0.3]
            for pos in slot_positions:
                geom_types.append("cylinder")
                geom_sizes.append([prototype_radius, prototype_half_length])
                geom_locations.append([pos[0], pos[1], prototype_center_z])
                if geom_rgbas is not None:
                    geom_rgbas.append(prototype_rgba)

        sites = []
        slot_rgba = slot_rgba if slot_rgba is not None else [0.15, 0.15, 0.15, 1.0]
        for idx, pos in enumerate(slot_positions):
            sites.append(
                {
                    "name": f"slot_{idx}",
                    "pos": list(pos),
                    "size": [slot_size],
                    "rgba": slot_rgba,
                }
            )

        super().__init__(
            name=name,
            total_size=list(base_half_size),
            geom_types=geom_types,
            geom_sizes=geom_sizes,
            geom_locations=geom_locations,
            geom_rgbas=geom_rgbas,
            joints=None,
            sites=sites,
            locations_relative_to_center=True,
        )
