#########################################
###  SkeletonPython Module for Bisque   ###
#########################################
import os
import time
import sys
import logging
import zipfile
from lxml import etree
import numpy as np
from optparse import OptionParser
from bqapi.comm import BQSession, BQCommError

from bqapi.util import fetch_dataset, fetch_image_pixels, d2xml
from subprocess import call

import pickle
import csv
import copy

#Constants
PARALLEL                        = True
NUMBER_OF_THREADS               = 4 #number of concurrent requests
IMAGE_SERVICE                   = 'image_service'
FEATURES_SERVICE                = 'features'
FEATURE_NAME                    = 'HTD'
FEATURE_TABLE_DIR               = 'Outputs'
TEMP_DIR                        = 'Temp'

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('bq.modules')


class SkeletonPythonError(Exception):
    
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message     
class SkeletonPython(object):
    """
        SkeletonPython Model
    """

    def mex_parameter_parser(self, mex_xml):
        """
            Parses input of the xml and add it to SkeletonPython Trainer's options attribute
            
            @param: mex_xml
        """
        mex_inputs = mex_xml.xpath('tag[@name="inputs"]')
        if mex_inputs:
            for tag in mex_inputs[0]:
                if tag.tag == 'tag' and tag.attrib['type'] != 'system-input':
                    log.debug('Set options with %s as %s'%(tag.attrib['name'],tag.attrib['value']))
                    setattr(self.options,tag.attrib['name'],tag.attrib['value'])
        else:
            log.debug('SkeletonPythonFS: No Inputs Found on MEX!')

    def validateInput(self):
        """
            Parses input of the xml and add it to SkeletonPython's options attribute
            
            @param: mex_xml
        """        
        if (self.options.mexURL and self.options.token): #run module through engine service
            return True
        
        if (self.options.user and self.options.pwd and self.options.root): #run module locally (note: to test module)
            return True
        
        log.debug('SkeletonPython: Insufficient options or arguments to start this module')
        return False


    def setup(self):
        """
            Fetches the mex, appends input_configurations to the option
            attribute of SkeletonPython and looks up the model on bisque to 
            classify the provided resource.
        """
        if (self.options.user and self.options.pwd and self.options.root):
            self.bqSession = BQSession().init_local( self.options.user, self.options.pwd, bisque_root=self.options.root)
            self.options.mexURL = self.bqSession.mex.uri
        # This is when the module actually runs on the server with a mexURL and an access token
        elif (self.options.mexURL and self.options.token):
            self.bqSession = BQSession().init_mex(self.options.mexURL, self.options.token)
        else:
            return
        
        # Parse the xml and construct the tree, also set options to proper values after parsing it (like image url)
        self.mex_parameter_parser(self.bqSession.mex.xmltree)
        
        log.debug('SkeletonPython: image URL: %s, mexURL: %s, stagingPath: %s, token: %s' % (self.options.image_url, self.options.mexURL, self.options.stagingPath, self.options.token))
    
    

    def construct_vertices(self, child):
	annotation_type = 'bg'
	if 'foreground' in child.values():
		annotation_type = 'fg'
	
	roi = []
	
	log.debug("This is the child")
	vertices = child.getchildren()[0].getchildren()
	for vertex in vertices:
		values = vertex.values()
		roi.append({'x':int(float(values[2])), 'y':int(float(values[3]))})
	self.rois[annotation_type].append(roi)
	log.debug(vertices)
	log.debug(len(vertices))

    def show_structure(self, r_xml):
	for i, child in enumerate(r_xml.getchildren()):
		if "background" in child.values() or 'foreground' in child.values():
			
			log.debug('Background/Foreground annotation')
			self.construct_vertices(child)
		else:
			self.show_structure(child)

    def run(self):
        """
            The core of the SkeletonPython Module
            
            Requests features on the image provided. Classifies each tile
            and picks a majority among the tiles. 
        """

	self.rois = {'fg':[],'bg':[]}

        r_xml = self.bqSession.fetchxml(self.options.mexURL, view='deep')
	log.debug("Shols structura")
	self.show_structure(r_xml)
	log.debug(self.rois)


        image = self.bqSession.load(self.options.image_url)
        ip = image.pixels().format('tiff')
        pixels = ip.fetch()
        f = open('./temp.tif','wb')
        f.write(pixels)
        f.close()
        
        pickle.dump([self.rois,self.options.segmentImage,self.options.deepNetworkChoice,self.options.qualitySeg,self.options.deepSeg,self.options.mexURL,self.options.token], open('./data.p','wb'))
        
        pathToScript = './DeepTools/deep_script.sh'
        call([pathToScript])

    def teardown(self):
        """
            Posting results to the mex
        """

        self.bqSession.update_mex('Returning results...')
        log.debug('Returning results...')
	log.debug(self.options.deepNetworkChoice)

        prediction = "None-Module Failure"

        
        outputTag = etree.Element('tag', name='outputs')
        outputSubTagImage = etree.SubElement(outputTag, 'tag', name='Final Image', value=self.options.image_url)
	

        print "Module will output image, {}".format(self.options.image_url)
    	
	if not os.path.isfile("./contours.pkl"):
		print "Module will not segment image, (were foreground and background polyline annotations provided?)"

	if self.options.segmentImage != "False" and os.path.isfile("./contours.pkl"):

		[contours, t_scale] = pickle.load(open("./contours.pkl","rb"))
		
		gob = etree.SubElement (outputSubTagImage, 'gobject', name='Annotations', type='Annotations')

		polyseg = etree.SubElement(gob, 'polygon', name='SEG')  
		etree.SubElement( polyseg, 'tag', name='color', value="#0000FF")
		opd = 0
		output_sampling = 1+int(len(contours)/100)
		for j in range(len(contours)):
			if j % (output_sampling) != -1:
				opd += 1
				etree.SubElement (polyseg, 'vertex', x=str(1+int(t_scale[1]*contours[j][1])), y=str(1+int(t_scale[0]*contours[j][0])))
		log.debug(opd)

	if self.options.deepNetworkChoice != 'None':
		outputSubTagSummary = etree.SubElement(outputTag, 'tag', name='summary')
		etree.SubElement(outputSubTagSummary, 'tag',name='Model File', value=self.options.deepNetworkChoice)
		etree.SubElement(outputSubTagSummary, 'tag',name='Segment Image', value=self.options.segmentImage)

	if self.options.deepNetworkChoice == "Simple classification":

		with open("./results.txt","r") as f:
		    for line in f:
		        if "PREDICTION_C:" in line:
		            prediction_c = line[14:-1]
		        if "CONFIDENCE_C:" in line:
		            confidence_c = line[14:-1]
		log.debug("IN here")
		classes = ["Leaf (PO:0025034): http://browser.planteome.org/amigo/term/PO:0025034","Fruit (PO:0009001): http://browser.planteome.org/amigo/term/PO:0009001","Flower (PO:0009046): http://browser.planteome.org/amigo/term/PO:0009046","Stem (PO:0009047): http://browser.planteome.org/amigo/term/PO:0009047","Whole plant (PO:0000003): http://browser.planteome.org/amigo/term/PO:0000003 "]
		if self.options.deepNetworkChoice != 'None':
			for i,class_tag in enumerate(classes):
				if str(i) in prediction_c:
					prediction_c = prediction_c.replace(str(i),class_tag)
					break
		etree.SubElement(outputSubTagSummary, 'tag',name='Class', value=str(prediction_c))
		link_direction = str(prediction_c).index("): ")+3
		etree.SubElement(outputSubTagSummary, 'tag',type='link',name='Class link', value=str(prediction_c)[link_direction:])
		etree.SubElement(outputSubTagSummary, 'tag', name='Class Confidence', value=str(confidence_c))
	elif self.options.deepNetworkChoice == "Leaf classification":
		leaf_targets = {'LeafType':["SIMPLE","COMPOUND","NONE"],'LeafShape':["ACEROSE","AWL-SHAPED","GLADIATE","HASTATE","CORDATE","DELTOID","LANCEOLATE","LINEAR","ELLIPTIC","ENSIFORM","LYRATE",
				   "OBCORDATE","FALCATE","FLABELLATE","OBDELTOID","OBELLIPTIC","OBLANCEOLATE","OBLONG","PERFOLIATE","QUADRATE","OBOVATE","ORBICULAR",
				   "RENIFORM","RHOMBIC","OVAL","OVATE","ROTUND","SAGITTATE","PANDURATE","PELTATE","SPATULATE","SUBULATE","NONE"],
				'Leafbaseshape':["AEQUILATERAL","ATTENUATE","AURICULATE","CORDATE","CUNEATE","HASTATE","OBLIQUE","ROUNDED","SAGITTATE","TRUNCATE","NONE"],
				'Leaftipshape': ["CIRROSE","CUSPIDATE","ACUMINATE","ACUTE","EMARGINATE","MUCRONATE","APICULATE","ARISTATE","MUCRONULATE","MUTICOUS","ARISTULATE","CAUDATE","OBCORDATE","OBTUSE","RETUSE","ROUNDED","SUBACUTE","TRUNCATE","NONE"]
				   ,'Leafmargin':["BIDENTATE","BIFID","DENTATE","DENTICULATE","BIPINNATIFID","BISERRATE","DIGITATE","DISSECTED","CLEFT","CRENATE","DIVIDED","ENTIRE","CRENULATE","CRISPED","EROSE","INCISED","INVOLUTE","LACERATE","PEDATE","PINNATIFID","LACINIATE","LOBED","PINNATILOBATE","PINNATISECT","LOBULATE","PALMATIFID","REPAND","REVOLUTE","PALMATISECT","PARTED","RUNCINATE","SERRATE","SERRULATE","SINUATE","TRIDENTATE","TRIFID","TRIPARTITE","TRIPINNATIFID","NONE"],'Leafvenation':["RETICULATE","PARALLEL","NONE"]}

		
		leaf_targets_links = copy.deepcopy(leaf_targets)
		for k in leaf_targets_links.keys():
			for j in range(len(leaf_targets_links[k])):
				leaf_targets_links[k][j] = 'undefined'
		
		# Read csv file from DeepTools
		# map each leaf target to the corresponding PO term
		# produce a link for the PO term
		# leaf_targets_links is now the same dictionary but with links instead of descriptions
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

				
			

		leaf_keys_proper_names  =['Leaf Type','Leaf Shape','Leaf Base Shape','Leaf Tip Shape','Leaf Margin','Leaf Venation']
		with open("./results.txt","r") as f:
			class_list = []
			for i,line in enumerate(f):
				leaf_keys  =['LeafType','LeafShape','Leafbaseshape','Leaftipshape','Leafmargin','Leafvenation']
				# Remove after introduction of the leaf classifier (below start with appends)
				if int(line) == len(leaf_targets[leaf_keys[i]])-1:
					line = '0'
				class_list.append(line)
				
				etree.SubElement(outputSubTagSummary, 'tag',name=leaf_keys_proper_names[i]+"-Name", value=str(leaf_targets[leaf_keys[i]][int(class_list[i])]))
				if str(leaf_targets_links[leaf_keys[i]][int(class_list[i])]) != 'undefined':
					etree.SubElement(outputSubTagSummary, 'tag',type='link',name=leaf_keys_proper_names[i]+'-Accession', value=str('http://browser.planteome.org/amigo/term/'+leaf_targets_links[leaf_keys[i]][int(class_list[i])]))
				else:
					etree.SubElement(outputSubTagSummary, 'tag',name=leaf_keys_proper_names[i]+'-Accession', value=str(leaf_targets_links[leaf_keys[i]][int(class_list[i])]))
					
		
            
        self.bqSession.finish_mex(tags = [outputTag])
        log.debug('FINISHED')
        self.bqSession.close()


    def main(self):
        """
            The main function that runs everything
        """

        print("DEBUG_INIT")

        log.debug('SkeletonPython is called with the following arguments')
        log.debug('sysargv : %s\n\n' % sys.argv )
    
        
        parser = OptionParser()

        parser.add_option( '--image_url'   , dest="image_url")
        parser.add_option( '--mex_url'     , dest="mexURL")
        parser.add_option( '--module_dir'  , dest="modulePath")
        parser.add_option( '--staging_path', dest="stagingPath")
        parser.add_option( '--bisque_token', dest="token")
        parser.add_option( '--user'        , dest="user")
        parser.add_option( '--pwd'         , dest="pwd")
        parser.add_option( '--root'        , dest="root")

        (options, args) = parser.parse_args()

        # Set up the mexURL and token based on the arguments passed to the script
        try: #pull out the mex
            log.debug("options %s" % options)
            if not options.mexURL:
                options.mexURL = sys.argv[1]
            if not options.token:
                options.token = sys.argv[2]
                
        except IndexError: #no argv were set
            pass
        
        if not options.stagingPath:
            options.stagingPath = ''
        
        # Still don't have an imgurl, but it will be set up in self.setup()
        log.debug('\n\nPARAMS : %s \n\n Options: %s'%(args, options))
        self.options = options
        
        if self.validateInput():
            
            try: #run setup and retrieve mex variables
                self.setup()
            except Exception, e:
                log.exception("Exception during setup")
                self.bqSession.fail_mex(msg = "Exception during setup: %s" %  str(e))
                return
            
            try: #run module operation
                self.run()
            except SkeletonPythonError, e:
                log.exception("Exception during run")
                self.bqSession.fail_mex(msg = "Exception during run: %s" % str(e.message))
                return                

            except Exception, e:
                log.exception("Exception during run")
                self.bqSession.fail_mex(msg = "Exception during run: %s" % str(e))
                return

            try: #post module
                self.teardown()
            except Exception, e:
                log.exception("Exception during teardown %s")
                self.bqSession.fail_mex(msg = "Exception during teardown: %s" %  str(e))
                return

if __name__ == "__main__":
    SkeletonPython().main()
    
    
