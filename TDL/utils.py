from transition import Transition
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib.pyplot as plt
from torch import Tensor
from os import listdir
import numpy as np
import mujoco_py
import glob
import time
import os


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def tsample(self):
        b = Transition(*zip(*self.memory))
        fileds = list(b)
        return [Tensor(item) for item in fileds]

    def __len__(self):
        return len(self.memory)


class Render(object):
    def __init__(self, sim, env_name, width=400, height=400, mp4size=(4, 4)):
        self.sim = sim
        self.context = mujoco_py.MjRenderContextOffscreen(sim, -1)
        self.width = width
        self.height = height
        self.record = []
        self.mp4size = mp4size
        self.env_name = env_name

    def _flush(self):
        self.record = []

    def render(self):
        self.record.append(self._render())
        
    def _render(self):
        if self.env_name in ['Ant-v2', 'Humanoid-v2', 'Humanoid-v2']:
            # 3D environments
            self.context.cam.lookat[0] = self.sim.data.qpos[0]
            self.context.cam.lookat[1] = self.sim.data.qpos[1]
        elif self.env_name in ['HalfCheetah-v2', 'Hopper-v2', 'Swimmer-v2', 'Walker2d-v2']:
            # 2D environments
            self.context.cam.lookat[0] = self.sim.data.qpos[0]
        self.context.render(self.width, self.height, -1)
        pixels, _ = self.context.read_pixels(self.width, self.height)
        return pixels

    def to_mp4(self, path):

        tic = time.time()
        num_frames = len(self.record)
        record = np.array(self.record)

        fig, ax = plt.subplots(figsize=self.mp4size)
        img = ax.imshow(record[0], origin='lower')
        ax.set_axis_off()
        
        def update(i):
            img.set_data(record[i])
            fig.canvas.draw()
            fig.canvas.flush_events()

        anim = FuncAnimation(fig, update, frames=np.arange(0, num_frames), interval=50)

        # search for lock
        FLAG_FILE = 'MP4_OUTPUT_FLAG'
        while FLAG_FILE in listdir('.'):
            time.sleep(10)
            print('.', end='', flush=True)

        # add a lock
        with open(FLAG_FILE, 'w') as f:
            f.write('')

        # prevent FFMPG crash
        try:
            anim.save(path, writer=animation.FFMpegFileWriter(fps=20))
        except OSError as e:
            pass
            
        for f in glob.glob('./_tmp*.png'):
            os.remove(f)
        plt.close()

        # cancel the lock
        os.remove(FLAG_FILE)

        toc = time.time()
        print('outputed {} frames using {}s'.format(num_frames, toc - tic))

        self._flush()


class Schduler(object):
    def __init__(self, init_val, schedule, total_steps, **kwargs):
        self.init_val = init_val
        self.total_steps = total_steps
        self.schedule = schedule
        self.counter = 0
        if schedule == 'none' or schedule == 'constant':
            self.value = self.constant_value
        elif schedule == 'linear':
            self.end_val = kwargs['end_val'] if 'end_val' in kwargs else 0
            self.value = self.linear_value
        elif schedule == 'half_linear':
            self.end_val = kwargs['end_val'] if 'end_val' in kwargs else 0
            self.value = self.half_linear_value
        elif schedule == 'exponential':
            self.end_val = kwargs['end_val'] if 'end_val' in kwargs else 0
            self.beta = kwargs['beta'] if 'beta' in kwargs else 5.0
            self.value = self.exponential_value

    def constant_value(self):
        return self.init_val

    def linear_value(self):
        if self.counter < self.total_steps:
            return self.end_val + (self.init_val - self.end_val) * (1 - self.counter / self.total_steps) 
        else:
            return self.end_val

    def half_linear_value(self):
        return self.end_val + (self.init_val - self.end_val) * min(2 * (1 - self.counter / self.total_steps), 1.0)

    def exponential_value(self):
        return self.end_val + (self.init_val - self.end_val) * np.exp( - self.beta * self.counter / self.total_steps)

    def step(self):
        self.counter += 1


def dict_to_object(dic):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return Struct(**dic)

def to_cuda(cid, *args):
    return [arg.cuda(cid) for arg in args]
