import numpy as np
from shapely import geometry
from fiona.crs import from_epsg
from fiona import collection
import fiona
import utm
import subprocess
from osgeo import gdal
import glob
import ogr
import ee
import random


def authenticate_get_code():
    get_ipython().magic('%bash')
    #earthengine authenticate --quiet
    return
    
def authenticate(code):
    get_ipython().magic('%bash')
    #earthengine authenticate --authorization-code=code
    
    try:
      ee.Initialize()
      print('The Earth Engine package initialized successfully!')
    except ee.EEException as e:
      print('The Earth Engine package failed to initialize!')
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    ee.Initialize()
    return

def sample(outfolder, shapereference, ownfile=False, centr=False):
    src = fiona.open(shapereference)
    schema = { 'geometry': 'Polygon', 'properties': { 'name': 'str' } }
    
    with collection(outfolder + "/some.geojson", "w", "GeoJSON", schema=schema) as output:
        for f in src:
            polygon = f['geometry']['coordinates'][0][0]#f[0]['geometry']['coordinates']
            P = geometry.Polygon([p for p in polygon])
            C = P.centroid
            U = utm.from_latlon(C.y, C.x)
            P = geometry.point.Point(U) 
            if centr:
                B = P.centroid.buffer(scl, cap_style=3) 
            else:
                B = P.representative_point().buffer(scl, cap_style=3)
            eh = geometry.mapping(B)
            wgsbuffs = []
            for i, hm in enumerate(eh['coordinates'][0]):
                try:
                    coords = utm.to_latlon(hm[0], hm[1], U[2], U[3])
                    wgsbuffs.append(coords)
                except:
                    print('ha')               

            if len(wgsbuffs) == 5:
                P2 = geometry.Polygon([(p[1],p[0]) for p in wgsbuffs])    
                output.write({
                    'properties': {
                        'name': 'bla'
                    },
                    'geometry': geometry.mapping(P2)
                    })

                newpoly = [[w[1],w[0]] for w in wgsbuffs]

                outraster = 'Y' + str(int(C.y*100)) + 'X' + str(int(C.x*100))
                
                if type(ownfile) == str:
                    ulx = np.min(np.array(newpoly),0)[0]
                    uly = np.max(np.array(newpoly),0)[1]
                    lrx = np.max(np.array(newpoly),0)[0]
                    lry = np.min(np.array(newpoly),0)[1]
                    subprocess.call(['gdal_translate', '-projwin', str(ulx), str(uly), str(lrx), str(lry), ownfile, outfolder + '/' + outraster + '.tif'])

                else:
                    
                    ####### something like, e.g.:
                    sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    vh = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filter(ee.Filter.eq('instrumentMode', 'IW'))
                    vh = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filter(ee.Filter.eq('instrumentMode', 'IW'))
                    vhDescending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

                    composite = ee.Image.cat([
                      vhDescending.select('VH').mean(),
                      ee.ImageCollection(vhDescending.select('VV').merge(vhDescending.select('VV'))).mean(),
                      vhDescending.select('VH').mean()
                    ]).focal_median()
                    ####################################                    
                    
                    
                    task = ee.batch.Export.image.toDrive(composite, 'hm/' + outraster, scale=10, region=newpoly)
                    task.start()
                    
def create_reference(working_folder, shapefile):
    rasterfilenames = glob.glob(working_folder + '/*.tif')
    Shp_src = ogr.Open(shapefile)
    rasterdriver = gdal.GetDriverByName('GTiff')

    for r in rasterfilenames:
        referencerasterloc = r
        outputname = r[:-4] + '_mask.tif'    
        Ras_src = gdal.Open(referencerasterloc)    

        new_raster = rasterdriver.Create(outputname, Ras_src.GetRasterBand(1).XSize, Ras_src.GetRasterBand(1).YSize, 1, gdal.GDT_Byte)
        new_raster.SetProjection(Ras_src.GetProjection())
        new_raster.SetGeoTransform(Ras_src.GetGeoTransform())

        Shp_lyr = Shp_src.GetLayer()
        gdal.RasterizeLayer(new_raster, [1], Shp_lyr, None, None, [1], ['ATTRIBUTE='+fieldname])
    return



def train_test_split(loc, p):
    y_files = glob.glob(loc + '/*mask.tif')
    x_files = [f[:-9] + '.tif' for f in y_files]#-6
    
    c = list(zip(y_files, x_files))
    random.shuffle(c)
    y_file, x_file = zip(*c)
    
    number = int(len(y_file) * p)
    
    x_file_tr = x_file[:number]
    y_file_tr = y_file[:number]
    
    x_file_te = x_file[number:]
    y_file_te = y_file[number:]
    
    return list(x_file_tr), list(y_file_tr), list(x_file_te), list(y_file_te)

def load_data_generate_batches(x_files_tr, y_files_tr, num_classes, bs=8, edgelength=32, augment=True, centr=True):
 
    Xss = []
    yss = []
    
    for i, x in enumerate(x_files_tr):
        
        # load xs
        S = gdal.Open(x)
        A = S.ReadAsArray(0, 0, edgelength, edgelength) # define edgelength       
        where_are_NaNs = np.isnan(A)
        A[where_are_NaNs] = -40.            
        X = np.transpose(A)
        
        S_y = gdal.Open(y_files_tr[i])        
        if centr == True:
            y = S_y.ReadAsArray(int(edgelength/2), int(edgelength/2), 1, 1)[0][0] # central pixel            
        else:
            y = S_y.ReadAsArray(0, 0, edgelength, edgelength) # define edgelength also; also think about NoData
            
            
        Xss.append(X)
        yss.append(y)
        
        if augment == True:
            for r in range(0,3):
                Xss.append(np.rot90(X, r))
                Xss.append(np.fliplr(np.rot90(X, r)))
                yss.append(y)
                yss.append(y)
                
    # shuffle both synchronously
    
    c = list(zip(yss, Xss))
    random.shuffle(c)
    yss, Xss = zip(*c)
    
    # then...
    X_batch = []
    y_batch = []
    X_batches = []
    y_batches = []
    
    yss = list(yss)
    Xss = list(Xss)
    
    for i, x in enumerate(Xss):
        X_batch.append(x)
        y_batch.append(yss[i])
        
        if (i % bs == 0) and (i != 0):
            if centr == True:
                y_corr = np.array(y_batch)
                y_corr[y_corr == 255] = 0
                n_values = num_classes
                y_batch = np.eye(n_values)[y_corr]
            
            X_batches.append(X_batch)
            y_batches.append(y_batch)
            
            X_batch = []
            y_batch = []
            
    X_batches[0] = X_batches[0][:-1]
    y_batches[0] = y_batches[0][:-1]
    
    return X_batches, y_batches   
    

def generate_test(X_locs, y_locs, edgelength, num_classes, centr=True):
    #print(enumerate(X_locs))
    Xss = []
    yss = []
    for i, x in enumerate(X_locs):
        S = gdal.Open(x)
        S_y = gdal.Open(y_locs[i])
        A = S.ReadAsArray(0, 0, edgelength, edgelength)
        where_are_NaNs = np.isnan(A)
        A[where_are_NaNs] = -40.
        A = np.transpose(A)
        
        if centr == True:
            y = S_y.ReadAsArray(int(edgelength/2), int(edgelength/2), 1, 1)[0][0]
        else:
            y = S_y.ReadAsArray(0, 0, edgelength, edgelength)
        
        Xss.append(A)
        yss.append(y)
    
    if centr == True:
        y_file = np.array(yss)
        #if np.min(y_file) < 0:
        #    print('hello')
        y_file[y_file == 255] = 0
        n_values = num_classes
        yss = np.eye(n_values)[y_file]
    return Xss, yss

def standardization_parameters(batches_of_x, test_x):
    X = np.array(batches_of_x)
    X_te = np.array(test_x)
    
    means = []
    sds = []
    
    for i in range(X.shape[4]):
        mean = np.mean(X[:,:,:,:,i])
        sd = np.std(X[:,:,:,:,i])
        X[:,:,:,:,i] = (X[:,:,:,:,i] - mean) / sd
        X_te[:,:,:,i] = (X_te[:,:,:,i] - mean) / sd
        means.append(mean)
        sds.append(sd)
        
    return list(X), list(X_te), means, sds

def batchload(WORKDIR, train_test_ratio, num_classes, batch_size=8, edgelength=32, centr=True):
    xtr, ytr, xte, yte = train_test_split(WORKDIR, train_test_ratio)
    X_batches, y_batches = load_data_generate_batches(xtr,ytr,num_classes, batch_size,edgelength,True,True)
    xte, yte = generate_test(xte,yte,edgelength,num_classes,True)
    X_batches, xte, MNs, SDs = standardization_parameters(X_batches, xte)
    return X_batches, y_batches, xte, yte, MNs, SDs