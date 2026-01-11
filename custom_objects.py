from robosuite.models.objects import CompositeObject


class PalletObject(CompositeObject):
    def __init__(
        self,
        name,
        base_half_size,
        slot_positions,
        base_rgba=None,
        slot_rgba=None,
        slot_size=0.006,
    ):
        geom_types = ["box"]
        geom_sizes = [list(base_half_size)]
        geom_locations = [[0.0, 0.0, 0.0]]
        geom_rgbas = [base_rgba] if base_rgba is not None else None

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
