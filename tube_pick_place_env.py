import numpy as np

from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.mjcf_utils import array_to_string

from custom_objects import PalletObject, TubeObject


def _build_slot_positions(base_half_size, rows=4, cols=3, margin=(0.02, 0.02), z_offset=0.002):
    x_span = max(base_half_size[0] * 2 - 2 * margin[0], 0.0)
    y_span = max(base_half_size[1] * 2 - 2 * margin[1], 0.0)
    x_spacing = x_span / (cols - 1) if cols > 1 else 0.0
    y_spacing = y_span / (rows - 1) if rows > 1 else 0.0
    x_start = -x_span / 2.0
    y_start = -y_span / 2.0

    positions = []
    for r in range(rows):
        for c in range(cols):
            x = x_start + c * x_spacing
            y = y_start + r * y_spacing
            z = base_half_size[2] + z_offset
            positions.append([x, y, z])
    return positions


class TubePickPlace(PickPlace):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0.0, 0.0, 0.8),
        bin1_pos=(0.0, -0.2, 0.8),
        pallet_xy=(0.2, 0.0),
        pallet_half_size=(0.08, 0.1, 0.015),
        slot_radius=None,
        tube_radius=0.01,
        tube_half_length=0.06,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=300,
        ignore_done=False,
        hard_reset=True,
        camera_names=("agentview", "robot0_eye_in_hand"),
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
    ):
        self.object_to_id = {"tube": 0}
        self.obj_names = ["Tube"]
        self.object_id = 0
        self.table_offset = np.array(table_offset)
        self.tube_radius = tube_radius
        self.tube_half_length = tube_half_length
        tube_length = 2.0 * self.tube_half_length
        self.hole_depth = tube_length * 0.75
        pallet_half_size = np.array(pallet_half_size, dtype=float)
        min_half_z = self.hole_depth / 2.0
        if pallet_half_size[2] < min_half_z:
            pallet_half_size = pallet_half_size.copy()
            pallet_half_size[2] = min_half_z
        self.pallet_half_size = pallet_half_size
        if slot_radius is None:
            slot_radius = self.tube_radius * 1.1
        self.slot_radius = slot_radius
        self.slot_positions = np.array(_build_slot_positions(self.pallet_half_size))
        self.pallet_pos = np.array(
            [pallet_xy[0], pallet_xy[1], self.table_offset[2] + self.pallet_half_size[2]]
        )
        bin1_pos = (bin1_pos[0], bin1_pos[1], self.table_offset[2])
        bin2_pos = (self.pallet_pos[0], self.pallet_pos[1], self.table_offset[2])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            table_full_size=table_full_size,
            table_friction=table_friction,
            bin1_pos=bin1_pos,
            bin2_pos=bin2_pos,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
            single_object_mode=2,
            object_type=None,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def _load_model(self):
        super(PickPlace, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.tube = TubeObject(
            name="Tube",
            radius=self.tube_radius,
            half_length=self.tube_half_length,
            rgba=[0.85, 0.1, 0.1, 1.0],
            include_cap=False,
        )

        self.pallet = PalletObject(
            name="Pallet",
            base_half_size=self.pallet_half_size,
            slot_positions=self.slot_positions,
            base_rgba=[0.1, 0.25, 0.7, 1.0],
            slot_rgba=[0.2, 0.2, 0.2, 1.0],
            slot_size=self.slot_radius,
            hole_radius=self.slot_radius,
            hole_depth=self.hole_depth,
            hole_rgba=[0.08, 0.08, 0.08, 1.0],
            prototype_radius=self.tube_radius,
            prototype_half_length=self.tube_half_length,
            prototype_rgba=[0.85, 0.1, 0.1, 0.3],
            prototype_center_z=self.pallet_half_size[2] - self.hole_depth + self.tube_half_length,
        )
        self.pallet.get_obj().set("pos", array_to_string(self.pallet_pos))

        self.objects = [self.tube]

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.pallet] + self.objects,
        )

        self._get_placement_initializer()

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="TubeSampler",
                mujoco_objects=self.objects,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation=None,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=0.01,
            )
        )

    def _setup_references(self):
        super(PickPlace, self)._setup_references()

        self.obj_body_id = {}
        self.obj_geom_id = {}
        for obj in self.objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        self.objects_in_bins = np.zeros(len(self.objects))
        self.bin_size = np.array([self.pallet_half_size[0] * 2, self.pallet_half_size[1] * 2, 0.0])
        self.target_bin_placements = np.zeros((len(self.objects), 3))
        for i in range(len(self.objects)):
            self.target_bin_placements[i, :] = [
                self.pallet_pos[0],
                self.pallet_pos[1],
                self.table_offset[2],
            ]

    def _reset_internal(self):
        super(PickPlace, self)._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        self.sim.model.body_pos[self.sim.model.body_name2id(self.pallet.root_body)] = self.pallet_pos

        obj_names = {obj.name for obj in self.objects}
        self.obj_to_use = self.objects[self.object_id].name
        if self.single_object_mode in {1, 2}:
            obj_names.remove(self.obj_to_use)
            self.clear_objects(list(obj_names))

        if self.single_object_mode != 0:
            for i, sensor_names in self.object_id_to_sensors.items():
                for name in sensor_names:
                    self._observables[name].set_enabled(i == self.object_id)
                    self._observables[name].set_active(i == self.object_id)

    def not_in_bin(self, obj_pos, bin_id):
        slot_positions_world = self.slot_positions + self.pallet_pos
        xy_dists = np.linalg.norm(slot_positions_world[:, :2] - obj_pos[:2], axis=1)
        pallet_top_z = self.pallet_pos[2] + self.pallet_half_size[2]
        in_slot = np.any(xy_dists <= self.slot_radius)
        in_height = pallet_top_z <= obj_pos[2] <= pallet_top_z + 0.15
        return not (in_slot and in_height)

    def staged_rewards(self):
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        active_objs = []
        for i, obj in enumerate(self.objects):
            if self.objects_in_bins[i]:
                continue
            active_objs.append(obj)

        r_reach = 0.0
        if active_objs:
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=active_obj.root_body,
                    target_type="body",
                    return_distance=True,
                )
                for active_obj in active_objs
            ]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        r_grasp = (
            int(
                self._check_grasp(
                    gripper=self.robots[0].gripper,
                    object_geoms=[g for active_obj in active_objs for g in active_obj.contact_geoms],
                )
            )
            * grasp_mult
        )

        r_lift = 0.0
        if active_objs and r_grasp > 0.0:
            z_target = self.bin2_pos[2] + 0.25
            object_z_locs = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name] for active_obj in active_objs]][
                :, 2
            ]
            z_dists = np.maximum(z_target - object_z_locs, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (lift_mult - grasp_mult)

        r_hover = 0.0
        if active_objs:
            slot_positions_world = self.slot_positions + self.pallet_pos
            object_xy_locs = self.sim.data.body_xpos[[self.obj_body_id[active_obj.name] for active_obj in active_objs]][
                :, :2
            ]
            dists = np.linalg.norm(slot_positions_world[:, :2][None, :, :] - object_xy_locs[:, None, :], axis=2)
            min_dists = np.min(dists, axis=1)
            objects_above_slots = min_dists < self.slot_radius
            r_hover_all = np.zeros(len(active_objs))
            r_hover_all[objects_above_slots] = lift_mult + (1 - np.tanh(10.0 * min_dists[objects_above_slots])) * (
                hover_mult - lift_mult
            )
            r_hover_all[~objects_above_slots] = r_lift + (1 - np.tanh(10.0 * min_dists[~objects_above_slots])) * (
                hover_mult - lift_mult
            )
            r_hover = np.max(r_hover_all)

        return r_reach, r_grasp, r_lift, r_hover
