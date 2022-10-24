import abc
from memory_profiler import profile
import warnings
import time
import glfw
from gym import error
from gym.utils import seeding
import numpy as np
from os import path
import gym
import sys

try:
    import mujoco_py

    print(mujoco_py.__file__)
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))


def _assert_task_is_set(func):
    def inner(*args, **kwargs):
        env = args[0]
        if not env._set_task_called:
            raise RuntimeError(
                'You must call env.set_task before using env.'
                + func.__name__
            )
        return func(*args, **kwargs)

    return inner


DEFAULT_SIZE = 500


class MujocoEnv(gym.Env, abc.ABC):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    max_path_length = 500

    def __init__(self, model_path, frame_skip):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)

        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._did_see_sim_exception = False

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abc.abstractmethod
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        pass

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @_assert_task_is_set
    def reset(self):
        self._did_see_sim_exception = False
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if getattr(self, 'curr_path_length', 0) > self.max_path_length:
            raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            try:
                self.sim.step()
            except mujoco_py.MujocoException as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    '''
    def render(self, offscreen=False, camera_name="corner2", resolution=(640, 480)):
    	print('Rendering')
        assert_string = ("camera_name should be one of ",
                "corner3, corner, corner2, topview, gripperPOV, behindGripper")
        assert camera_name in {"corner3", "corner", "corner2", 
            "topview", "gripperPOV", "behindGripper"}, assert_string
        if not offscreen:
            self._get_viewer('human').render()
        else:
            return self.sim.render(
                *resolution,
                mode='offscreen',
                camera_name=camera_name
            )
    
    '''

    def render(self, offscreen=False, camera_name="corner2", resolution=(640, 480), mode='human'):
        assert_string = ("camera_name should be one of ", "corner3, corner, corner2, topview, gripperPOV, behindGripper")
        assert camera_name in {"corner3", "corner", "corner2", "topview", "gripperPOV", "behindGripper"}, assert_string
        if not offscreen:
            if mode == 'human':
                #self._get_viewer('human').render()
                return self.sim.render(
                    *resolution,
                    mode='offscreen',
                    camera_name=camera_name
                )
            else:
                '''
                self._get_viewer('depth_array').render(resolution[0], resolution[1])
                data_tridimensional = self._get_viewer('depth_array').read_pixels(resolution[0], resolution[1], depth=True, segmentation=False)[0]
                data_array = self._get_viewer('depth_array').read_pixels(resolution[0], resolution[1], depth=True, segmentation=False)[1]
                print(np.shape(data_tridimensional))
                return data_tridimensional
                '''

                render_context = mujoco_py.MjRenderContextOffscreen(self.sim)
                render_context.render(width=resolution[0], height=resolution[1], camera_id=self.model.camera_name2id(camera_name), segmentation=False)
                render_image_depth = render_context.read_pixels(resolution[0], resolution[1], depth=True, segmentation=False) # 0: Imagen Normal 128x128x3, 1: Array Profundidad 128x128x1
                del render_context
                '''

                (mujoco_py.MjRenderContextOffscreen(self.sim)).render(width=resolution[0], height=resolution[1], camera_id=self.model.camera_name2id(camera_name), segmentation=False)
                return mujoco_py.MjRenderContextOffscreen(self.sim).read_pixels(resolution[0], resolution[1], depth=True, segmentation=False)
                '''
                #print('Size of mujoco_py OBJECT: {}'.format(sys.getsizeof(mujoco_py)))
                #print('Size of self OBJECT: {}'.format(sys.getsizeof(self)))
                #print('Size of render_image_depth OBJECT: {}'.format(sys.getsizeof(render_image_depth)))

                '''
                try:
                    print('Size of render_context OBJECT: {}'.format(sys.getsizeof(render_context)))
                except:
                    print('No render_context')
                '''
                    
                return render_image_depth
        else:
            return self.sim.render(
                *resolution,
                mode='offscreen',
                camera_name=camera_name
            )

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None
    '''
    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer
    '''

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)
