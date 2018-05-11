
# +
# future compatability code
# -
from __future__ import print_function


# +
# import(s)
# -
import logging
import os
import pickle
import sys
import copy
import csv

try:
    # noinspection PyPep8Naming
    from lxml import etree as eTree
except ImportError:
    import xml.etree.ElementTree as eTree

from bqapi.comm import BQSession
from optparse import OptionParser
from subprocess import call


# +
# constant(s)
# -
MODULE_NAME = 'PlanteomeDeepSegment'
MODULE_VERSION = 'v0.2.0'
MODULE_DATE = '1 May, 2018'
MODULE_AUTHORS = 'Dimitrios Trigkakis, Justin Preece, Blake Joyce, Phil Daly'
MODULE_DESCRIPTION = '{} Module for BisQue {}'.format(MODULE_NAME, MODULE_VERSION)
MODULE_SOURCE = '{}.py'.format(MODULE_NAME)

PICKLE_CONTOURS_FILE = 'contours.pkl'
PICKLE_DATA_FILE = 'data.p'
TEXT_RESULTS_FILE = 'results.txt'
TIFF_IMAGE_FILE = 'temp.tif'


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('bq.modules')


# +
# class: PlanteomeDeepSegmentError()
# -
class PlanteomeDeepSegmentError(Exception):

    def __init__(self, errstr=''):
        self.errstr = errstr


# +
# class: PlanteomeDeepSegment()
# -
class PlanteomeDeepSegment(object):

    # +
    # __init__()
    # -
    def __init__(self):

        # entry message
        log.debug('{}.__init__()> message on entry'.format(MODULE_NAME))

        # declare some variables and initialize them
        self.options = None
        self.bqSession = None
        self.rois = None
        self.message = None

        # get full path(s) to file(s)
        self.contours_file = os.path.abspath(os.path.expanduser(PICKLE_CONTOURS_FILE))
        self.data_file = os.path.abspath(os.path.expanduser(PICKLE_DATA_FILE))
        self.results_file = os.path.abspath(os.path.expanduser(TEXT_RESULTS_FILE))
        self.tiff_file = os.path.abspath(os.path.expanduser(TIFF_IMAGE_FILE))

        log.debug('{}.__init()> self.contours_file={}'.format(MODULE_NAME, self.contours_file))
        log.debug('{}.__init()> self.data_file={}'.format(MODULE_NAME, self.data_file))
        log.debug('{}.__init()> self.results_file={}'.format(MODULE_NAME, self.results_file))
        log.debug('{}.__init()> self.tiff_file={}'.format(MODULE_NAME, self.tiff_file))

        # exit message
        log.debug('{}.__init__()> message on exit'.format(MODULE_NAME))

    # +
    # hidden method: _mex_parameter_parser()
    # -
    def _mex_parameter_parser(self, mex_xml=None):

        # entry message
        log.debug('{}._mex_parameter_parser()> message on entry, mex_xml={}'.format(MODULE_NAME, str(mex_xml)))

        if mex_xml is not None:
            mex_inputs = mex_xml.xpath('tag[@name="inputs"]')
            if mex_inputs:
                for tag in mex_inputs[0]:
                    if tag.tag == 'tag' and tag.attrib['type'] != 'system-input':
                        _name = tag.attrib['name'].strip()
                        _value = tag.attrib['value'].strip()
                        log.debug('{}._mex_parameter_parser()> setting self.options.{}={}'.format(
                            MODULE_NAME, _name, _value))
                        setattr(self.options, _name, _value)
                        log.debug("{}._mex_parameter_parser()> set self.options.{}={}".format(
                            MODULE_NAME, _name, getattr(self.options, _name)))
            else:
                log.error('{}.mex_parameter_parser()> no inputs found on mex!'.format(MODULE_NAME))
        else:
            self.message = '{}.mex_parameter_parser()> mex_xml is None'.format(MODULE_NAME)
            log.error(self.message)

        # exit message
        log.debug('{}.main()> message on exit, options={}'.format(MODULE_NAME, self.options))

    # +
    # hidden method: _validate_input()
    # -
    def _validate_input(self):

        # entry message
        retval = False
        log.debug('{}._validate_input()> message on entry, retval={}'.format(MODULE_NAME, retval))

        # run module through engine_service (default)
        if self.options.mexURL and self.options.token:
            retval = True

        # run module locally
        elif self.options.user and self.options.pwd and self.options.root:
            retval = True

        else:
            retval = False
            log.error('{}.validate_input()> insufficient options or arguments to start this module'.format(MODULE_NAME))

        # exit message
        log.debug('{}._validate_input()> message on exit, retval={}'.format(MODULE_NAME, retval))
        return retval

    # +
    # hidden method: _construct_vertices()
    # -
    def _construct_vertices(self, child=None):

        # entry message
        vertices = None
        roi = []
        log.debug('{}._construct_vertices()> message on entry, child={}'.format(MODULE_NAME, str(child)))

        # get annotation type
        if child is not None:
            annotation_type = 'fg' if 'foreground' in child.values() else 'bg'

            # get vertices
            vertices = child.getchildren()[0].getchildren()
            for _vertex in vertices:
                _values = _vertex.values()
                roi.append({'x': int(float(_values[2])), 'y': int(float(_values[3]))})
            self.rois[annotation_type].append(roi)

        # exit message
        log.debug('{}._construct_vertices()> message on exit, vertices={}, length={}'.format(
            MODULE_NAME, str(vertices), len(vertices)))

    # +
    # hidden method: _show_structure()
    # -
    def _show_structure(self, r_xml=None):

        # entry message
        log.debug('{}._show_structure()> message on entry, r_xml={}'.format(MODULE_NAME, str(r_xml)))

        if r_xml is not None:
            for _i, _child in enumerate(r_xml.getchildren()):
                if 'background' in _child.values() or 'foreground' in _child.values():
                    self._construct_vertices(_child)
                else:
                    self._show_structure(_child)

        # exit message
        log.debug('{}._show_structure()> message on exit'.format(MODULE_NAME))

    # +
    # method: setup()
    # -
    def setup(self):

        # entry message
        log.debug('{}.setup()> message on entry, options={}'.format(MODULE_NAME, self.options))

        # run locally
        if self.options.user and self.options.pwd and self.options.root:
            log.debug('{}.setup()> running locally with user={}, pwd={}, root={}'.format(
                MODULE_NAME, self.options.user, self.options.pwd, self.options.root))
            self.bqSession = BQSession().init_local(self.options.user, self.options.pwd, bisque_root=self.options.root)
            self.options.mexURL = self.bqSession.mex.uri

        # run on the server with a mexURL and an access token
        elif self.options.mexURL and self.options.token:
            log.debug('{}.setup()> running on server with mexURL={}, token={}'.format(
                MODULE_NAME, self.options.mexURL, self.options.token))
            self.bqSession = BQSession().init_mex(self.options.mexURL, self.options.token)

        # failed to connect to bisque
        else:
            self.bqSession = None
            self.message('{}.setup()> failed to connect to bisque'.format(MODULE_NAME))
            log.error(self.message)
            raise PlanteomeDeepSegmentError(self.message)

        # parse the xml and construct the tree, also set options to proper values after parsing it
        if self.bqSession is not None:
            self._mex_parameter_parser(self.bqSession.mex.xmltree)
            log.debug('{}.setup()> image URL={}, mexURL={}, stagingPath={}, token={}'.format(
                MODULE_NAME, self.options.image_url, self.options.mexURL, self.options.stagingPath, self.options.token))

        # exit message
        log.debug('{}.setup()> message on exit, options={}'.format(MODULE_NAME, self.options))

    # +
    # method: run()
    # The core of the PlanteomeDeepSegment module. It requests features on the provided image, classifies each tile
    # and selects a majority amongst the tiles.
    # -
    def run(self):

        # entry message
        log.debug('{}.run()> message on entry, options={}'.format(MODULE_NAME, self.options))

        self.rois = {'fg': [], 'bg': []}
        r_xml = self.bqSession.fetchxml(self.options.mexURL, view='deep')
        log.debug('{}.run()> Shols structura'.format(MODULE_NAME))
        self._show_structure(r_xml)

        # +
        # << START OF TEMPORARY DEBUG MEASURE FOR TESTING OF FLOWER.JPG ON BRONN.CYVERSE.ORG ONLY!! >>
        # -
        log.debug("TEMPORARY DATA STRUCTURE!!")
        if (not self.rois.get('bg', [])) and (not self.rois.get('fg', [])):
            self.rois = {
                'bg': [[{'y': 110, 'x': 341}, {'y': 113, 'x': 347}, {'y': 113, 'x': 353}, {'y': 116, 'x': 358},
                        {'y': 117, 'x': 363}, {'y': 120, 'x': 367}, {'y': 123, 'x': 375}, {'y': 131, 'x': 387},
                        {'y': 136, 'x': 391}, {'y': 141, 'x': 397}, {'y': 145, 'x': 399}, {'y': 157, 'x': 414},
                        {'y': 176, 'x': 430}, {'y': 185, 'x': 434}, {'y': 208, 'x': 439}, {'y': 280, 'x': 439},
                        {'y': 290, 'x': 435}, {'y': 297, 'x': 431}, {'y': 301, 'x': 430}, {'y': 303, 'x': 427},
                        {'y': 305, 'x': 427}, {'y': 318, 'x': 411}, {'y': 326, 'x': 390}, {'y': 332, 'x': 379},
                        {'y': 341, 'x': 350}, {'y': 347, 'x': 299}, {'y': 348, 'x': 294}, {'y': 349, 'x': 294},
                        {'y': 349, 'x': 287}, {'y': 341, 'x': 274}, {'y': 336, 'x': 263}, {'y': 317, 'x': 238},
                        {'y': 303, 'x': 223}, {'y': 290, 'x': 215}, {'y': 271, 'x': 205}, {'y': 250, 'x': 199},
                        {'y': 200, 'x': 199}, {'y': 194, 'x': 200}, {'y': 170, 'x': 213}, {'y': 158, 'x': 223},
                        {'y': 142, 'x': 239}, {'y': 132, 'x': 256}, {'y': 126, 'x': 270}, {'y': 120, 'x': 296},
                        {'y': 120, 'x': 301}, {'y': 119, 'x': 301}]],
                'fg': [[{'y': 193, 'x': 364}, {'y': 193, 'x': 371}, {'y': 198, 'x': 375}, {'y': 218, 'x': 383},
                        {'y': 232, 'x': 385}, {'y': 264, 'x': 385}, {'y': 270, 'x': 383}, {'y': 273, 'x': 381},
                        {'y': 276, 'x': 363}, {'y': 276, 'x': 324}, {'y': 271, 'x': 315}, {'y': 264, 'x': 308},
                        {'y': 261, 'x': 307}, {'y': 260, 'x': 305}, {'y': 255, 'x': 301}, {'y': 252, 'x': 299},
                        {'y': 249, 'x': 299}, {'y': 246, 'x': 297}, {'y': 237, 'x': 294}, {'y': 233, 'x': 294},
                        {'y': 227, 'x': 291}, {'y': 194, 'x': 291}, {'y': 185, 'x': 297}, {'y': 185, 'x': 299},
                        {'y': 183, 'x': 299}, {'y': 182, 'x': 300}]]
            }
        # +
        # << END OF TEMPORARY DEBUG MEASURE FOR TESTING OF FLOWER.JPG ON BRONN.CYVERSE.ORG ONLY!! >>
        # -
        log.debug(self.rois)

        # dump image as .tiff
        image = self.bqSession.load(self.options.image_url)
        ip = image.pixels().format('tiff')
        # pixels = ip.fetch()
        with open(self.tiff_file, 'wb') as f:
            f.write(ip.fetch())

        # pickle the data
        try:
            if self.rois and getattr(self.options, 'segmentImage') != '' and \
                    getattr(self.options, 'deepNetworkChoice') != '' and getattr(self.options, 'qualitySeg') != '' and \
                    getattr(self.options, 'deepSeg') != '' and getattr(self.options, 'mexURL') != '' and \
                    getattr(self.options, 'token') != '':
                log.debug('{}.run()> pickling data to {}'.format(
                    MODULE_NAME, self.data_file))
                pickle.dump([self.rois, self.options.segmentImage, self.options.deepNetworkChoice,
                             self.options.qualitySeg, self.options.deepSeg, self.options.mexURL, self.options.token],
                            open(self.data_file, 'wb'))
        except AttributeError as e:
            self.message('{}.run()> failed to pickle data, e={}'.format(MODULE_NAME, str(e)))
            log.error(self.message)

        # 00TODO: replace with class!!
        path_to_script = './DeepTools/deep_script.sh'
        call([path_to_script])

        # exit message
        log.debug('{}.run()> message on exit, options={}'.format(MODULE_NAME, self.options))

    # +
    # method: teardown()
    # -
    def teardown(self):

        # entry message
        log.debug('{}.teardown()> message on entry, options={}'.format(MODULE_NAME, self.options))
        self.bqSession.update_mex('Returning results...')

        print('Module will output image, {}'.format(self.options.image_url))

        # segment the image (if required)

        output_tag = eTree.Element('tag', name='outputs')
        output_sub_tag_image = eTree.SubElement(output_tag, 'tag', name='Final Image', value=self.options.image_url)

        if self.options.segmentImage.lower() == 'true' and os.path.isfile(self.contours_file):
            log.debug('{}.teardown()> module will segment image from file {}'.format(MODULE_NAME, self.contours_file))

            [_contours, _t_scale] = pickle.load(open(self.contours_file, 'rb'))
            _gob = eTree.SubElement(output_sub_tag_image, 'gobject', name='Annotations', type='Annotations')
            _polyseg = eTree.SubElement(_gob, 'polygon', name='SEG')
            eTree.SubElement(_polyseg, 'tag', name='color', value="#0000FF")
            _opd = 0
            _output_sampling = 1 + int(len(_contours)/100)
            for _j in range(len(_contours)):
                if _j % _output_sampling == 0:
                    _opd += 1
                    _x = str(1 + int(_t_scale[1]*_contours[_j][1]))
                    _y = str(1 + int(_t_scale[0]*_contours[_j][0]))
                    eTree.SubElement(_polyseg, 'vertex', x=_x, y=_y)
            log.debug('{}.teardown()> _opd={}'.format(MODULE_NAME, _opd))
        else:
            print('Module will not segment image, (were foreground and background polyline annotations provided?)')

        # set tag(s)
        if getattr(self.options, 'deepNetworkChoice', '') != '' and self.options.deepNetworkChoice.lower() != 'none':
            if self.options.deepNetworkChoice.lower() == 'simple classification':
                # get prediction and confidence
                prediction_c = -1
                confidence_c = 0.0
                prediction_t = -1
                confidence_t = 0.0
                try:
                    with open(self.results_file, 'r') as f:
                        for _line in f:
                            if _line.strip() != '':
                                log.debug('{}.teardown()> _line={}'.format(MODULE_NAME, _line))
                                if 'PREDICTION_C:' in _line:
                                    prediction_c = int(_line.split(':')[1].strip())
                                if 'CONFIDENCE_C:' in _line:
                                    confidence_c = float(_line.split(':')[1].strip())
                                if 'PREDICTION_T:' in _line:
                                    prediction_t = int(_line.split(':')[1].strip())
                                if 'CONFIDENCE_T:' in _line:
                                    confidence_t = float(_line.split(':')[1].strip())
                except IOError as e:
                    self.message = '{}.teardown()> io error reading results, e={}'.format(MODULE_NAME, str(e))
                    log.error(self.message)
                finally:
                    log.debug('{}.teardown()> prediction_c={}'.format(MODULE_NAME, prediction_c))
                    log.debug('{}.teardown()> confidence_c={}'.format(MODULE_NAME, confidence_c))
                    log.debug('{}.teardown()> prediction_t={}'.format(MODULE_NAME, prediction_t))
                    log.debug('{}.teardown()> confidence_t={}'.format(MODULE_NAME, confidence_t))

                # annotate with prediction
                classes = ["Leaf (PO:0025034): http://browser.planteome.org/amigo/term/PO:0025034","Fruit (PO:0009001): http://browser.planteome.org/amigo/term/PO:0009001","Flower (PO:0009046): http://browser.planteome.org/amigo/term/PO:0009046","Stem (PO:0009047): http://browser.planteome.org/amigo/term/PO:0009047","Whole plant (PO:0000003): http://browser.planteome.org/amigo/term/PO:0000003 "]

                prediction_c = classes[prediction_c] if (0 <= prediction_c <= len(classes)) else 'unknown'

                output_sub_tag_summary = eTree.SubElement(output_tag, 'tag', name='summary')

                link_direction = str(prediction_c).index("): ")+3

                eTree.SubElement(output_sub_tag_summary, 'tag', name='Model File', value=self.options.deepNetworkChoice)
                if getattr(self.options, 'segmentImage', '') != '':
                    eTree.SubElement(output_sub_tag_summary, 'tag', name='Segment Image', value=self.options.segmentImage)
                eTree.SubElement(output_sub_tag_summary, 'tag', name='Class', value=prediction_c)
                eTree.SubElement(output_sub_tag_summary, 'tag',type='link',name='Accession', value=str(prediction_c)[link_direction:])
                eTree.SubElement(output_sub_tag_summary, 'tag', name='Class Confidence', value=str(confidence_c))
            
            if self.options.deepNetworkChoice.lower() == 'leaf classification':
                leaf_targets = {'LeafType':["SIMPLE","COMPOUND","NONE"],'LeafShape':["ACEROSE","AWL-SHAPED","GLADIATE","HASTATE","CORDATE","DELTOID","LANCEOLATE","LINEAR","ELLIPTIC","ENSIFORM","LYRATE",
				   "OBCORDATE","FALCATE","FLABELLATE","OBDELTOID","OBELLIPTIC","OBLANCEOLATE","OBLONG","PERFOLIATE","QUADRATE","OBOVATE","ORBICULAR",
				   "RENIFORM","RHOMBIC","OVAL","OVATE","ROTUND","SAGITTATE","PANDURATE","PELTATE","SPATULATE","SUBULATE","NONE"],
				'Leafbaseshape':["AEQUILATERAL","ATTENUATE","AURICULATE","CORDATE","CUNEATE","HASTATE","OBLIQUE","ROUNDED","SAGITTATE","TRUNCATE","NONE"],
				'Leaftipshape': ["CIRROSE","CUSPIDATE","ACUMINATE","ACUTE","EMARGINATE","MUCRONATE","APICULATE","ARISTATE","MUCRONULATE","MUTICOUS","ARISTULATE","CAUDATE","OBCORDATE","OBTUSE","RETUSE","ROUNDED","SUBACUTE","TRUNCATE","NONE"]
				   ,'Leafmargin':["BIDENTATE","BIFID","DENTATE","DENTICULATE","BIPINNATIFID","BISERRATE","DIGITATE","DISSECTED","CLEFT","CRENATE","DIVIDED","ENTIRE","CRENULATE","CRISPED","EROSE","INCISED","INVOLUTE","LACERATE","PEDATE","PINNATIFID","LACINIATE","LOBED","PINNATILOBATE","PINNATISECT","LOBULATE","PALMATIFID","REPAND","REVOLUTE","PALMATISECT","PARTED","RUNCINATE","SERRATE","SERRULATE","SINUATE","TRIDENTATE","TRIFID","TRIPARTITE","TRIPINNATIFID","NONE"],'Leafvenation':["RETICULATE","PARALLEL","NONE"]}

        
                leaf_keys_proper_names  =['Leaf Type','Leaf Shape','Leaf Base Shape','Leaf Tip Shape','Leaf Margin','Leaf Venation']
                leaf_keys  =['LeafType','LeafShape','Leafbaseshape','Leaftipshape','Leafmargin','Leafvenation']
                leaf_targets_links = copy.deepcopy(leaf_targets)
                for k in leaf_targets_links.keys():
                    for j in range(len(leaf_targets_links[k])):
                        leaf_targets_links[k][j] = 'undefined'
        
                with open("./LeafMappings.csv") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',',quotechar='|')
            
                    current_name = ''
                    for row in reader:
                        name = row[0]
                        po_term = row[1]
                        if po_term == '':
                            po_term = "undefined"
                        for leaf_category in leaf_targets_links.keys():
                            if name.replace(" ","").lower() == leaf_category.replace(" ","").lower():
                                current_name = leaf_category
                                break
                        if current_name in leaf_targets.keys(): 
                            for leaf_category in leaf_targets[current_name]:
                                if name.replace(" ","").lower() == leaf_category.replace(" ","").lower():
                                    i = leaf_targets[current_name].index(leaf_category)
                                    leaf_targets_links[current_name][i] = po_term
                                    break           

                output_sub_tag_summary = eTree.SubElement(output_tag, 'tag', name='summary')
                eTree.SubElement(output_sub_tag_summary, 'tag',name='Model File', value=self.options.deepNetworkChoice)
                eTree.SubElement(output_sub_tag_summary, 'tag',name='Segment Image', value=self.options.segmentImage)
                with open("./results.txt","r") as f:
                    class_list = []
                    for i, line in enumerate(f):
                        log.debug("i {}, line {}".format(i, line))
                        # Remove after introduction of the leaf classifier (below start with appends)
                        if int(line) == len(leaf_targets[leaf_keys[i]])-1:
                            line = '0'
                        class_list.append(line)
                        
                                      
                        eTree.SubElement(output_sub_tag_summary, 'tag',name=leaf_keys_proper_names[i]+"-Name", value=str(leaf_targets[leaf_keys[i]][int(class_list[i])]))
                        if str(leaf_targets_links[leaf_keys[i]][int(class_list[i])]) != 'undefined':
                            eTree.SubElement(output_sub_tag_summary, 'tag',type='link',name=leaf_keys_proper_names[i]+'-Accession', value=str('http://browser.planteome.org/amigo/term/'+leaf_targets_links[leaf_keys[i]][int(class_list[i])]))
                        else:
                            eTree.SubElement(output_sub_tag_summary, 'tag',name=leaf_keys_proper_names[i]+'-Accession', value=str(leaf_targets_links[leaf_keys[i]][int(class_list[i])]))                                                                                   

        self.bqSession.finish_mex(tags=[output_tag])
        self.bqSession.close()

        # exit message
        log.debug('{}.teardown()> message on exit, options={}'.format(MODULE_NAME, self.options))

    # +
    # method: main()
    # -
    def main(self):

        # entry message
        log.debug('{}.main()> message on entry, args={}'.format(MODULE_NAME, sys.argv))

        parser = OptionParser()
        parser.add_option('--image_url', dest="image_url")
        parser.add_option('--mex_url', dest="mexURL")
        parser.add_option('--module_dir', dest="modulePath")
        parser.add_option('--staging_path', dest="stagingPath")
        parser.add_option('--bisque_token', dest="token")
        parser.add_option('--user', dest="user")
        parser.add_option('--pwd', dest="pwd")
        parser.add_option('--root', dest="root")
        (options, args) = parser.parse_args()

        log.debug('{}.main()> options={}'.format(MODULE_NAME, options))
        log.debug('{}.main()> args={}'.format(MODULE_NAME, args))

        # set up the mexURL and token based on the arguments passed to the script
        try:
            if not options.mexURL:
                options.mexURL = sys.argv[1]
            if not options.token:
                options.token = sys.argv[2]
            if not options.stagingPath:
                options.stagingPath = ''
        except IndexError:
            pass
        finally:
            self.options = options
            log.debug('{}.main()> self.options={}'.format(MODULE_NAME, self.options))

        # check input(s)
        if self._validate_input():

            # noinspection PyBroadException
            try:
                # set up the module
                self.setup()
            except PlanteomeDeepSegmentError as e:
                self.message = '{}.main()> specific exception after setup(), e={}'.format(MODULE_NAME, str(e.errstr))
                log.exception(self.message)
                self.bqSession.fail_mex(msg=self.message)
                raise PlanteomeDeepSegmentError(self.message)
            except Exception as e:
                self.message = '{}.main()> exception after setup(), e={}'.format(MODULE_NAME, str(e))
                log.exception(self.message)
                self.bqSession.fail_mex(msg=self.message)
                raise PlanteomeDeepSegmentError(self.message)
            except:
                self.message = '{}.main()> error after setup()'.format(MODULE_NAME)
                log.exception(self.message)
                self.bqSession.fail_mex(msg=self.message)
                raise PlanteomeDeepSegmentError(self.message)

            # noinspection PyBroadException
            try:
                # run the module
                self.run()
            except PlanteomeDeepSegmentError as e:
                self.message = '{}.main()> specific exception after run(), e={}'.format(MODULE_NAME, str(e.errstr))
                log.exception(self.message)
                self.bqSession.fail_mex(msg=self.message)
                raise PlanteomeDeepSegmentError(self.message)
            except Exception as e:
                self.message = '{}.main()> exception after run(), e={}'.format(MODULE_NAME, str(e))
                log.exception(self.message)
                self.bqSession.fail_mex(msg=self.message)
                raise PlanteomeDeepSegmentError(self.message)
            except:
                self.message = '{}.main()> error after run()'.format(MODULE_NAME)
                log.exception(self.message)
                self.bqSession.fail_mex(msg=self.message)
                raise PlanteomeDeepSegmentError(self.message)

            # noinspection PyBroadException
            try:
                # tear down the module
                self.teardown()
            except PlanteomeDeepSegmentError as e:
                self.message = '{}.main()> specific exception after teardown(), e={}'.format(MODULE_NAME, str(e.errstr))
                log.exception(self.message)
                self.bqSession.fail_mex(msg=self.message)
                raise PlanteomeDeepSegmentError(self.message)
            except Exception as e:
                self.message = '{}.main()> exception after teardown(), e={}'.format(MODULE_NAME, str(e))
                log.exception(self.message)
                self.bqSession.fail_mex(msg=self.message)
                raise PlanteomeDeepSegmentError(self.message)
            except:
                self.message = '{}.main()> error after teardown()'.format(MODULE_NAME)
                log.exception(self.message)
                self.bqSession.fail_mex(msg=self.message)
                raise PlanteomeDeepSegmentError(self.message)

        else:
            self.message = '{}.main()> failed to validate instance'.format(MODULE_NAME)
            log.error(self.message)
            self.bqSession.fail_mex(msg=self.message)
            raise PlanteomeDeepSegmentError(self.message)

        # exit message
        log.debug('{}.main()> message on exit, args={}'.format(MODULE_NAME, sys.argv))


# +
# main()
# -
if __name__ == "__main__":
    try:
        log.debug('{}.__main__()> starting ...'.format(MODULE_NAME))
        P = PlanteomeDeepSegment()
        P.main()
        log.debug('{}.__main__()> done'.format(MODULE_NAME))
    except PlanteomeDeepSegmentError as err:
        print('{}.__main__()> failed, error={}'.format(MODULE_NAME, err.errstr))
