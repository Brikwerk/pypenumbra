import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pypenumbra as ppen
import pypenumbra.simulate as psim


class SimulationAnimation(animation.TimedAnimation):
    def __init__(self, kernel_size, point_dist, circle_radius):
        """Initialization of the matplotlib animation.
        
        :param kernel_size: The width/height of the kernel used
        :type kernel_size: int
        :param point_dist: The distance (in pixels) between the two point sources
        :type point_dist: int
        :param circle_radius: The radius of the blank penumbra circle
        :type circle_radius: int
        """
        self.kernel_size = kernel_size
        self.point_dist = point_dist
        self.circle_radius = circle_radius

        fig = plt.figure(figsize=(8, 8), dpi=200)
        self.ax1 = fig.add_subplot(321)
        self.ax1.axis('off')
        self.ax2 = fig.add_subplot(322)
        self.ax2.axis('off')
        self.ax3 = fig.add_subplot(312)
        self.ax3.axis('off')
        self.ax4 = fig.add_subplot(313)
        self.ax4.axis('off')
        fig.tight_layout(pad=4)

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        kernel = psim.create_dual_point_kernel(self.kernel_size, i)
        blank_penumbra = psim.generate_blank_penumbra_cr18x24(self.circle_radius)
        penumbra_img = psim.generate_penumbra(blank_penumbra, kernel)
        reconstruction_img, sinogram = ppen.reconstruct_from_array(penumbra_img)

        self.ax1.set_title("Kernel Image %dpx dist" % int(i))
        self.ax1.imshow(kernel, animated=True, cmap="gray")
        self.ax2.set_title("Penumbra Image")
        self.ax2.imshow(penumbra_img, animated=True, cmap="gray")
        self.ax3.set_title("Sinogram")
        self.ax3.imshow(sinogram, animated=True, cmap="gray")
        self.ax4.set_title("Reconstruction")
        self.ax4.imshow(reconstruction_img, animated=True, cmap="gray")

    def new_frame_seq(self):
        return iter(range(1, self.point_dist, 2))


ani = SimulationAnimation(159, 157, 100)
#ani.save('simulation.mp4')
plt.show()