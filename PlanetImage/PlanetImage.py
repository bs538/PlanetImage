import gdal
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io, exposure
from xml.dom import minidom
from copy import deepcopy

class FileOrganizer(object):
    """
    simple functions for organizing satellite images
    """
    def __init__(self,datapath,asset_type,item):
        self.datapath = datapath
        self.item = item
        self.asset_type = asset_type
        self.flist = os.listdir(datapath)
        self.flist.sort()
        self._create_sublists()
        self._get_dates()

        assert self.asset_type == 'analytic', 'other assets not yet implemented'
        assert self.item in ['PSScene4Band','PSOrthoTile'], 'other items not yet implemented' 
        
    def _create_sublists(self):
        self.flist_analytic = [f for f in self.flist if (f.endswith('.tif') and not '_udm_' in f)]
        self.flist_metadata = [f for f in self.flist if f.endswith('.xml')]
        self.flist_mask = [f for f in self.flist if (f.endswith('.tif') and '_udm_' in f)]

        for ff in [self.flist_mask,self.flist_metadata]:
            assert len(ff) == len(self.flist_analytic), 'metadata or mask files missing?'
    
    def _get_dates(self):
        if self.item == 'PSScene4Band':
            self.dates = [f.split('_')[0] for f in self.flist_analytic]
        elif self.item == 'PSOrthoTile':
            self.dates = [f.split('_')[2] for f in self.flist_analytic]
        else:
            raise NotImplementedError('unknown item type, cannot get dates')

    def select_by_namestring(self,string):
        """
        selects only images with given string in the filename
        """
        new = deepcopy(self)
        new.flist = [f for f in new.flist if string in f]
        new._create_sublists()
        new._get_dates()
        return new

    def rgb_preview(self,n=None,**rgb_kwargs):
        """
        reads in images and plots rgb view
        Todo: make this efficient for larger images
        """
        n = len(self.flist_analytic) if n is None else n
        for i,f in enumerate(self.flist_analytic[:n]):
            print(i,f)
            planetimg = PlanetImage(os.path.join(self.datapath,f),self.asset_type,self.item)
            planetimg.show_rgb(**rgb_kwargs)
            plt.show()




class PlanetImage(object):
    """
    simple class for Planet Geotiff images
    """
    def __init__(self,fpath,asset_type,item,
                 channel_order = ('B','G','R','NIR'),verbose=True):

        self.fpath = fpath
        self.asset_type = asset_type
        self.item = item
        self.channel_order = channel_order
        self.verbose = verbose    
        self._read_image()
        self._parse_metadata()
        if verbose:
            self.pprint()
        
        assert self.asset_type == 'analytic', 'other assets not yet implemented'
        assert self.item in ['PSScene4Band','PSOrthoTile'], 'other items not yet implemented' 
        
    def _read_image(self):
        """
        reads in image data using GDAL
        """
        self.ds = gdal.Open(self.fpath)
        self.data = self.ds.ReadAsArray()
        self.nbands = self.data.shape[0]
        self.gt = self.ds.GetGeoTransform()
        self.xpixsize = self.gt[1]
        self.ypixsize = abs(self.gt[5])

    def _find_metadata_and_mask(self):
        """
        creates filename of metadata and mask from path and checks if it is present in the same folder
        """
        assert 'clip' in self.fpath, 'currently only for data from clipping API'
        paths = self.fpath.split('/')
        fname = paths[-1]
        if self.item == 'PSScene4Band':
            splitstr = '_AnalyticMS'
            dummy = ''
        elif self.item == 'PSOrthoTile':
            splitstr = '_BGRN'
            dummy = '_Analytic'
        root = fname.split(splitstr)[0]
        fname_metadata = root+splitstr+dummy+'_metadata_clip.xml'
        fname_mask = root+splitstr+'_DN_udm_clip.tif'
        fpath_metadata = os.path.join('/',*paths[:-1],fname_metadata)
        fpath_mask = os.path.join('/',*paths[:-1],fname_mask)
        assert os.path.isfile(fpath_metadata), 'metadata not found in the same folder'
        assert os.path.isfile(fpath_mask), 'mask not found in the same folder'
        self.fpath_metadata = fpath_metadata
        self.fpath_mask = fpath_mask


    def _parse_metadata(self):
        """
        extracts interesting quantities from XML metadata
        adapted from https://www.planet.com/docs/guides/quickstart-ndvi/
        """
        self._find_metadata_and_mask()
        xmldoc = minidom.parse(self.fpath_metadata)
        
        #coefficients for conversion to TOA
        nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
        # XML parser refers to bands by numbers 1-4
        coeffs = {}
        for node in nodes:
            bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
            if bn in ['1', '2', '3', '4']:
                i = int(bn)
                value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                coeffs[i] = float(value)
        self.TOAcoeffs = coeffs

        #cloud cover
        self.cloudCoverPercentage = float(xmldoc.getElementsByTagName("opt:cloudCoverPercentage")[0].firstChild.data)

        #unusable data
        self.unusableDataPercentage = float(xmldoc.getElementsByTagName("ps:unusableDataPercentage")[0].firstChild.data)
        
    def pprint(self):
        """
        prints a summary of image properties
        """
        print('------------------------------------------------------------------------')        
        print('array shape =',self.data.shape)
        print('min,max = {:3.2e},{:3.2e}'.format(self.data.min(),self.data.max()))
        print('data type =',self.data.dtype)
        print('pixel size (x,y) = {},{} m'.format(self.xpixsize,self.ypixsize))
        print('bands :',self.channel_order)
        print('cloud cover percentage = {} percent'.format(self.cloudCoverPercentage))
        print('unusable data percentage = {} percent'.format(self.unusableDataPercentage))
        print('------------------------------------------------------------------------')
              
    def _TOA_check(self,use_TOA):
        """
        helper function to return appropriate attribute name for ToA or scaled radiance
        we can then access it coveniently by self.__dict__[dataname]
        """
        if use_TOA:
            dataname = 'TOAreflectance'
            if not hasattr(self,dataname):
                self.scale_to_TOA()
            print('using TOA reflectance')
        else:
            dataname = 'data'
            print('using scaled radiance')
        return dataname

    def show_hist(self,use_TOA = False,
                  hist_colors = {'B':'b','G':'g','R':'r','NIR':'blueviolet'}):
        """
        plots a histogram of the individual bands
        """

        dataname = self._TOA_check(use_TOA)

        fig,ax = plt.subplots()
        for d,c in zip(self.__dict__[dataname],self.channel_order):
            _ = ax.hist(d.ravel(),bins=100,histtype='step',
                         color=hist_colors[c],label=c)
        if use_TOA:
            label = 'TOA reflectance'
        else:
            label = 'scaled radiance'
        ax.set_xlabel(label)
        ax.legend()



    def show_rgb(self,use_TOA=False,
        simple=False,vmin=None,vmax=None,title=None,figsize=(6, 6)):
        """
        simple wrapper around plot function
        """
        rgbdata = self.get_rgbdata_mpl(use_TOA=use_TOA)
        PlanetImage.plot_rgb(rgbdata,simple=simple,vmin=vmin,vmax=vmax,title=title,figsize=figsize)
              
    def get_rgbdata_mpl(self,use_TOA=False,transpose_order=(1,2,0)):
        """
        checks if RGB channels present and extracts RGB data in the format expected by matplotlib
        arguments:
            transpose_order: transpose from (nbands,nypix,nxpix) to (nypix,nxpix,bands)
        """
        rgb = ['R','G','B']
        rgbdata = np.transpose(self.get_bands(rgb,use_TOA=use_TOA),transpose_order) 
        return rgbdata
        
    def get_bands(self,bands,use_TOA=False):
        """
        picks a subset of bands
        """
        assert sum([self.channel_order.count(c) for c in bands]) == len(bands), 'channels not found'
        bands_perm = [self.channel_order.index(c) for c in bands]
        
        dataname = self._TOA_check(use_TOA)
        return self.__dict__[dataname][bands_perm,:,:]

    def get_mask(self,show=True):
        """
        gets unusable data mask and computes masked percentage.
        assuming: 0 --> good data, 1--> no coverage, > 1 -->cloud
        """
        if not hasattr(self,'fpath_mask'):
            self._find_metadata_and_mask()
        
        self.mask = gdal.Open(self.fpath_mask).ReadAsArray()

        self.maskFractionClip = (self.mask > 0.).sum()/self.mask.size
        print('masked fraction in clip: {:3.1f} percent'.format(100*self.maskFractionClip))

        if show:
            plt.figure()
            plt.imshow(self.mask,vmin=0,vmax=2)
            plt.colorbar()

    def get_histnorm_rgb(self,use_TOA=False,lims=None,show=False):
        """
        produces histogram normalized RGB image
        uses skimage; 
        adapted from https://github.com/HyperionAnalytics/PyDataNYC2014/blob/master/color_image_processing.ipynb
        
        arguments:
            use_TOA (bool)
            lims: [(r_min,r_max), (g_min,g_max), (b_min,b_max)]
            show (bool): plot image
        """  
        rgbdata = self.get_rgbdata_mpl(use_TOA=use_TOA)

        #auto-setting limits
        if lims is None:
            lims = []
            for i in range(3):
                lims.append((rgbdata[:,:,i].min(),rgbdata[:,:,i].max()))
            print('auto-set limits to:',lims)

        rgb_ha = np.empty(rgbdata.shape, dtype=rgbdata.dtype)
        for lim, channel in zip(lims, range(3)):
            rgb_ha[:, :, channel] = exposure.rescale_intensity(rgbdata[:, :, channel], lim)
            
        if show:
            PlanetImage.plot_rgb(rgb_ha,simple=False)
            
        return rgb_ha   
    
    def plot_channels(self,use_TOA=False,**imshow_kwargs):
        """
        plots all bands on same colour scale
        all keywords are passed on to plt.imshow, 
        with special treatment for vmin,vmax
        """

        dataname = self._TOA_check(use_TOA)

        if not 'vmin' in imshow_kwargs:
            imshow_kwargs['vmin'] = self.__dict__[dataname].min()
        if not 'vmax' in imshow_kwargs:
            imshow_kwargs['vmax'] = self.__dict__[dataname].max()
            
        fig,axes = plt.subplots(1,self.nbands,figsize=(4*self.nbands,4))
        fig.subplots_adjust(wspace=0.01,hspace=0.01)
        for i,d in enumerate(self.__dict__[dataname]):
            im = axes[i].imshow(d,**imshow_kwargs)
            axes[i].set_title(self.channel_order[i])
            if i>0:
                axes[i].set_yticklabels([])
            if i<(self.nbands-1):
                #hacky way to shrink to same scale
                plt.colorbar(im,ax=axes[i],shrink=0.7).remove()
            else:
                plt.colorbar(im,ax=axes[i],shrink=0.8)
                
    
    def scale_to_TOA(self):
        """
        converts scaled radiance to ToA reflectance using coefficients from metadata
        """
        assert self.asset_type == 'analytic', 'tested only for asset type analytic'
        print('scaling to TOA with coeffs',self.TOAcoeffs)
        
        self.TOAreflectance = np.zeros(self.data.shape)
        for i in range(self.nbands):
            #coeffs is dict with keys starting at 1
            self.TOAreflectance[i] = self.data[i]*self.TOAcoeffs[i+1]
        
        return self.TOAreflectance
    
                
    def get_NDVI(self,use_TOA=True,show=False,lims=(None,None)):
        """
        calculates NDVI
        """
        R,NIR = self.get_bands(['R','NIR'],use_TOA=use_TOA)
        R = R.astype('float64') #necessary to avoid "underflow" of uint16
        NIR = NIR.astype('float64')
        with np.errstate(divide='ignore',invalid='ignore'): #suppress warnings
            ndvi = (NIR-R)/(NIR+R)
            
        if show:
            plt.imshow(ndvi,vmin=lims[0],vmax=lims[1])
            plt.colorbar(shrink=0.7)

        return ndvi
        
    
    @staticmethod
    def plot_rgb(image,simple=False,vmin=None,vmax=None,title=None,figsize=(6,6)):
        """
        plots RGB image
        arguments:
            image: 16bit or float (0-1)image in format (nypix,nxpix,3)
            simple (bool, only if 16 bit): plot image as is (no clipping etc.)
            ...
        """
        fig = plt.figure(figsize=figsize)
        fig.set_facecolor('white')


        if image.dtype == 'uint16':    
            if simple: #simple rescaling to 0-1
                plt.imshow(image/65535,interpolation='None')
            
            else: #clipping and converting to 8bit
                if vmin is None:
                    vmin = image.min()
                if vmax is None:
                    vmax = image.max()
                im = PlanetImage.lut_display(image,vmin,vmax)
                plt.imshow(im,interpolation='None')
        else:
            plt.imshow(image,interpolation='None')
            
        if title is not None:
            plt.title(title) 
            
            
    @staticmethod
    def lut_display(image, display_min, display_max):
        """
        helper function to clip a 16 bit RGB image and convert it to 8 bit for plotting 
        this version speeds up the plotting with a look-up-table and then uses "display" below
        copied from https://stackoverflow.com/questions/14464449/using-numpy-to-efficiently-convert-16-bit-image-data-to-8-bit-for-display-with
        """  
        lut = np.arange(2**16, dtype='uint16')
        lut = PlanetImage.display(lut, display_min, display_max)
        return np.take(lut, image)
            
    @staticmethod   
    def display(image, display_min, display_max): 
        """
        helper function to clip a 16 bit RGB image and convert it to 8 bit for plotting 
        copied from https://stackoverflow.com/questions/14464449/using-numpy-to-efficiently-convert-16-bit-image-data-to-8-bit-for-display-with
        """    
        # Here I set copy=True in order to ensure the original image is not
        # modified. If you don't mind modifying the original image, you can
        # set copy=False or skip this step.
        image = np.array(image, copy=True)
        image.clip(display_min, display_max, out=image)
        image -= display_min
        np.floor_divide(image, (display_max - display_min + 1) / 256,
                        out=image, casting='unsafe')
        return image.astype(np.uint8)