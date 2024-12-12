import boundary_evaluation as he
import cv2 as cv


def load_image(image_path, frame_thickness = 60):
    image = cv.imread(image_path)
    #gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image[frame_thickness:-frame_thickness, frame_thickness:-frame_thickness]

path = './models/pers_hom/homo_randomforest_border.joblib'

#error img
image_path = '../Data/scans/err_3faeden/doc00510820240703103910_001.jpg'
img = load_image(image_path, frame_thickness = 200)
evaluator = he.HomologyDetetor(path)
result = evaluator.analyse(img)

#good img
image_path = '../Data/scans/OK/doc00433320240524163153_001.jpg'
img = load_image(image_path, frame_thickness = 200)
#cv.imwrite('full.png', img)
evaluator = he.HomologyDetetor(path)
result = evaluator.analyse(img)

