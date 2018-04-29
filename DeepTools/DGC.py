import numpy as np
import maxflow
import cv2
import matplotlib.pyplot as plt
from skimage import measure

class logger(object):
    @staticmethod
    def log_levels():
        return ['Full', 'Semi', 'None']

    def __init__(self, log_level='None'):
        assert log_level in logger.log_levels()
        self.log_level = log_level

    def log(self, object):
        if self.log_level == 'Full':
            print(object)

class polyline(object):

    def __init__(self, vertices=None):  # vertices is a list of point dictionaries
        if vertices is not None:
            self.vertices = vertices
        else:
            self.vertices = []

        self.points = []

    def transform(self, scaling):
        for vertex in self.vertices:
            vertex['x'] *= scaling['w']
            vertex['y'] *= scaling['h']

    def bresenham_point(self, v_prev, v_curr, mask):
        x1, x2, y1, y2 = int(round(v_prev['x'])), int(round(v_curr['x'])), int(round(v_prev['y'])), int(round(v_curr['y']))
        dx, dy = x2 - x1, y2 - y1
        steep = abs(dy) > abs(dx)

        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        dx, dy = x2 - x1, y2 - y1

        error = int(dx / 2.0)
        y_add = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            point = (y, x) if steep else (x, y)
            mask[point[1], point[0]] = 255
            # self.points.append(point)
            error -= abs(dy)
            if error < 0:
                y += y_add
                error += dx

    def bresenham(self, mask):

        self.points = []

        vertex_prev = self.vertices[0]
        for i, vertex in enumerate(self.vertices, start=1):
            v_curr = vertex
            polypoints = self.bresenham_point(vertex_prev, v_curr, mask)
            vertex_prev = vertex

        return

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def del_vertex(self):
        self.vertices.pop()


class GraphCutter(object):

    def __init__(self, show_figures=True):  # Production-Fix : show=False
        self.show_figures = show_figures

    def show(self, image_data, name):
	return
        assert image_data is not None
        if not self.options['show_figures']:
            return

        if type(image_data) is not np.ndarray:
            if len(image_data.shape) == 3:
                image_data = image_data.permute(2,0,1).numpy()

        elif type(image_data) is np.ndarray:
            if len(image_data.shape) == 2:
                image_data = image_data[:,::-1]
            #image_data = np.swapaxes(image_data, 0, 2)

        plt.imsave(name, image_data[...,::-1])

    def logger(self,object):
        if self.options['log']:
            print(object)

    def lattice_constructor(self, connectivity=8, lattice_size=(0, 0)):

        r = lattice_size[0]
        c = lattice_size[1]

        if connectivity == 8 and r != 0 and c != 0:
            N = r * c
            M = r * (c - 1) + (r - 1) * c + 2 * (r - 1) * (c - 1)
            edges = np.zeros((M, 2), dtype=np.int32)
            edge_nodes = np.arange(N)
            edge_nodes = edge_nodes.reshape((r, c))

            Mtemp = r * (c - 1)
            edges[0:Mtemp, :] = \
                np.concatenate((edge_nodes[:, 0:c - 1].reshape((r * (c - 1), 1)), edge_nodes[:, 1:c].reshape((r * (c - 1), 1))), axis=1)

            edges[Mtemp:Mtemp + (r - 1) * c, :] = \
                np.concatenate((edge_nodes[0:r - 1, :].reshape(((r - 1) * c, 1)), edge_nodes[1:r, :].reshape(((r - 1) * c, 1))), axis=1)

            Mtemp = Mtemp + (r - 1) * c

            edges[Mtemp:Mtemp + (r - 1) * (c - 1), :] = \
                np.concatenate((edge_nodes[0:r - 1, 0:c - 1].reshape((r - 1) * (c - 1), 1), edge_nodes[1:r, 1:c].reshape(((r - 1) * (c - 1), 1))), axis=1)

            Mtemp = Mtemp + (r - 1) * (c - 1)

            edges[Mtemp:Mtemp + (r - 1) * (c - 1), :] = \
                np.concatenate((edge_nodes[0:r - 1, 1:c].reshape((r - 1) * (c - 1), 1), edge_nodes[1:r, 0:c - 1].reshape(((r - 1) * (c - 1), 1))), axis=1)

            return edges

    def edge_weights(self, parameters={"edges": None, "image": None, "lattice_size": (0, 0, 0), "nlink_sigma": 0}):

        r, c = parameters["lattice_size"]
        X, Y = np.meshgrid(np.arange(0, c), np.arange(0, r))
        edges = parameters["edges"]
        I = parameters["image"]

        self.show(I,"I.jpg")
        self.show(X,"X.jpg")
        self.show(Y,"Y.jpg")

        X0 = X[np.remainder(edges[:, 0], r).astype(np.int32), np.multiply(edges[:, 0], 1.0 / (r)).astype(np.int32)]
        X1 = X[np.remainder(edges[:, 1], r).astype(np.int32), np.multiply(edges[:, 1], 1.0 / (r)).astype(np.int32)]
        X_star = np.power(np.subtract(X0, X1), 2)

        Y0 = Y[np.remainder(edges[:, 0], r).astype(np.int32), np.multiply(edges[:, 0], 1.0 / (r)).astype(np.int32)]
        Y1 = Y[np.remainder(edges[:, 1], r).astype(np.int32), np.multiply(edges[:, 1], 1.0 / (r)).astype(np.int32)]
        Y_star = np.power(np.subtract(Y0, Y1), 2)

        euclidiean_distance = np.sqrt(np.add(X_star, Y_star))

        I0 = I[np.remainder(edges[:, 0], r).astype(np.int32),np.multiply(edges[:, 0], 1.0 / (r)).astype(np.int32)].astype(np.float32)
        I1 = I[np.remainder(edges[:, 1], r).astype(np.int32),np.multiply(edges[:, 1], 1.0 / (r)).astype(np.int32)].astype(np.float32)

        k = 2
        #print(I[0,0])
        #print(I0[0], I1[0])
        w_feat = np.power(np.abs(np.subtract(I0, I1)), k)
        self.show(w_feat.reshape(-1,1),"wfeat.jpg")

        sigma = parameters["nlink_sigma"]
        if parameters["nlink_sigma"] <= 0:
            sigma = np.sqrt(np.mean(w_feat))

        weights = np.multiply(np.exp(np.multiply(-1.0 / (2 * sigma ** 2), w_feat)),np.divide(1, euclidiean_distance))

        return weights, euclidiean_distance

    def vrl_gc(self, sizes, fg, bg, edges, weights):

        #print(sizes, fg.shape, bg.shape)
        grid_size = sizes[1] * sizes[2]
        g = maxflow.Graph[float](grid_size, (2 * grid_size) + (2 * sizes[3]))

        nodeids = g.add_nodes(grid_size)

        for i in range(grid_size):
            g.add_tedge(i, fg[int(i % sizes[1]), int(i / sizes[1])], bg[int(i % sizes[1]), int(i / sizes[1])])

        for i in range(sizes[3]):
            g.add_edge(edges[i, 0], edges[i, 1], weights[i], weights[i])

        flows = g.maxflow()
        output = np.zeros((sizes[1], sizes[2]))
        for i in range(len(nodeids)):
            output[int(i % sizes[1]), int(i / sizes[1])] = g.get_segment(nodeids[i])
        #print(g.get_segment(nodeids[0]))
        return output

        #return np.add(np.multiply(np.ones((sizes[1], sizes[2])), 1 / 4.0), np.random.rand(sizes[1], sizes[2]))

    def graph(
            self, polylines={},
            parameters={'bins': 8, 'sigma': 7.0, 'interaction_cost': 50, 'stroke_dt': True, 'stroke_var': 50,'hard_seeds': True},\
            options={'file': None, 'image_rgb': None, 'resize': False,'w': 0, 'h': 0, 'blur': True, 'show_figures': True, 'log': False, 'debug': False}):

        log = None
        if options['log']:
            self.logger = logger('Full')
            log = self.logger

        self.polylines = polylines
        self.options = options
        self.parameters = parameters

        if options['file'] is not None:
            self.I = cv2.imread(options['file'], 1)
        else:
            self.I = options['image_rgb']  # Debug-Fix : is it normalized to 0-1 and w-h-c?

        o_h, o_w, o_c = self.I.shape
        assert o_c == 3
        self.scaling = {'w': 1.0, 'h': 1.0}

        self.show(self.I,"OriginalImage.jpg")

        if options['resize']:
            resized_image = cv2.resize(self.I, (options['w'], options['h']))
            self.scaling = {'w': float(options['w']) / o_w, 'h': float(options['h']) / o_h}
            self.show(resized_image,"ResizedImage.jpg")
        else:
            resized_image = self.I
            options['w'] = self.I.shape[1]
            options['h'] = self.I.shape[0]

        # log.log("h x w x c --- input image {} transformed to {} with scaling {:0.2f}, {:0.2f}".format(self.I.shape, (
        # options['h'], options['w'], 3), *self.scaling.values()))

        blur = resized_image[:,:,2]
        self.show(blur,"BlurredImage_beforeGauss.jpg")
        blur = cv2.GaussianBlur(blur, (5, 5), 1.4)

        # polyline transformation to produce numpy pixel masks
        np_bg_mask = np.zeros((options['h'], options['w']), np.uint8)
        np_fg_mask = np.zeros((options['h'], options['w']), np.uint8)

        for loc in polylines.keys():
            mask = np_bg_mask if loc == 'bg' else np_fg_mask
            for polyline in polylines[loc]:
                polyline.transform(self.scaling)
                polyline.bresenham(mask)

        self.show(np_fg_mask,"FGmask.jpg")
        self.show(np_bg_mask,"BGmask.jpg")
        self.show(blur,"BlurredImage.jpg")

        histograms = {}
        dst = {}
        #print("entering polyline keys")
        for loc in polylines.keys():
            mask = np_bg_mask if loc == 'bg' else np_fg_mask
            channels = [0, 1, 2]
            ranges = [0, 256, 0 ,256, 0, 256]
            histograms[loc] = cv2.calcHist([resized_image], channels, mask,[parameters['bins'] for i in channels], ranges)
            histograms[loc] = histograms[loc] / np.sum(histograms[loc])
            b, g, r = cv2.split(resized_image / (256.0 / parameters['bins']))
            dst[loc] = np.float32(histograms[loc][b.astype(np.uint64).ravel(), g.astype(np.uint64).ravel(), r.astype(np.uint64).ravel()])
            dst[loc] = dst[loc].reshape(resized_image.shape[:2])
            self.show(dst[loc],"back-projection"+str(loc)+".jpg")
            #print("Sum for {} is {}".format(loc, np.sum(dst[loc])))

        if parameters['stroke_dt']:
            #print(np.max(np_fg_mask))
            dist_transform = cv2.distanceTransform(255-np_fg_mask, cv2.DIST_L2, maskSize=3)
            #dist_transform2 = cv2.distanceTransform(np_bg_mask, cv2.DIST_L2, maskSize=3)
            strokeDT = np.exp(-dist_transform / (parameters['stroke_var']*1.0))
            #strokeDT2 = np.exp(-dist_transform2 / (parameters['stroke_var']* 1.0))
            #print(np.min(strokeDT))
            #print(np.max(strokeDT))
            dst['fg'] = dst['fg'] * (strokeDT)
            dst['bg'] = dst['bg'] * (1-strokeDT)
            self.show(dst['fg'],"back-projection"+str('fg_DT_')+".jpg")
            self.show(dst['bg'],"back-projection"+str('bg_DT_')+".jpg")
            #print("Sum for {} is {}".format('fg', np.sum(dst['fg'])))

        for loc in polylines.keys():
            dst[loc] = -np.log(dst[loc]+0.01)

        if parameters['hard_seeds']:
            for loc in polylines.keys():
                mask = np_bg_mask if loc == 'fg' else np_fg_mask  # masks are inverted here
                dst[loc] = np.where(mask > 0, 10 ** 6, dst[loc])
                #print(mask.shape, np.max(mask), dst[loc].shape, np.max(dst[loc]), np.min(dst[loc]), np.mean(dst[loc]),
                #     np.median(dst[loc]))

        # edges is a numpy array of all node pairs in edges of a lattice of size w x h

        lattice_size = (options['w'], options['h'])
        edges = self.lattice_constructor(lattice_size=lattice_size)

        edge_parameters = {"edges": edges, "image": blur, "lattice_size": lattice_size[::-1], "nlink_sigma": parameters['sigma']}
        weights, w_dist = self.edge_weights(edge_parameters) # Image should be inverted to previous state, DT transform, hard seeds
        weights = np.expand_dims(weights, axis=1)

        #(weights)
        # set up all parameters to create the graph algorithm
        sizes = [2, options['h'], options['w'], edges.shape[0]]
        #print(dst['fg'].shape)
        seg_fg = self.vrl_gc(sizes, dst['fg'], dst['bg'], np.subtract(edges, 1), np.multiply(weights, parameters["interaction_cost"]))

        zero_padding = 1
        seg_fg[:zero_padding,:] = 0
        seg_fg[:,:zero_padding] = 0
        seg_fg[-zero_padding:,:] = 0
        seg_fg[:,-zero_padding:] = 0

        self.show(seg_fg,"segmentation.jpg")

        #seg_fg = seg_fg[::-1,:]
        seg = measure.find_contours(seg_fg, 0.8)
        self.show(seg_fg,"segmentation_f.jpg")

        max_points = 0
        m = None
        try:
            test_try = seg[0][0]
            for i in range(len(seg)):
                if len(seg[i]) > max_points:
                    max_points = len(seg[i])
                    m = seg[i]
        except:
            m = seg
        seg = m
        seg = seg[::,::]
	'''
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(self.I)
        ax.plot(seg[:,1], self.I.shape[0]-seg[:,0], linewidth=2)

        ax.axis('image')
        plt.show()
        fig.savefig("./figure.jpg")
	'''

        return {'segmentation':seg_fg,'contours':seg, 'polylines':polylines, 'options': options, 'parameters':parameters}

'''
GC = GraphCutter()
my_polylines = {'bg': [], 'fg': []}
my_options = {'file': "sample_input.jpg", 'image': None, 'resize': True, 'w': 481, 'h': 321, 'o_w': 481, 'o_h': 321, 'blur': True,
              'show_figures': True, 'debug': True, 'log': True}
my_parameters = {'bins': 8, 'sigma': 7.0, 'interaction_cost': 50, 'stroke_dt': True, 'hard_seeds': True, 'stroke_var': 50}

# Sample cuts, coordinates are 0,0 top-left corner, transformed with tw, th
my_polylines['bg'].append(polyline([{'x': 2, 'y': 2}, {'x': 10, 'y': 10}]))
my_polylines['fg'].append(polyline([{'x': 201, 'y': 160}, {'x': 281, 'y': 160}]))

out = GC.graph(polylines=my_polylines, options=my_options, parameters=my_parameters)
'''
