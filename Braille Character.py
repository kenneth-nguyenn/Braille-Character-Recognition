import cv2
import numpy as np
import imutils

from PIL import ImageFont, ImageDraw, Image # Custom font Vietnammese

# image = cv2.imread("shapes_and_colors.jpg")
image = cv2.imread("braille.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:2]
gray = gray[10:h-10, 15:w]

# morphological closing
# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
# div = np.float32(gray)/(close)
# res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))

# cv2.imshow("Final", res)
# cv2.imshow("close", close)
# cv2.imshow("Image", image)

# res2 = cv2.Canny(res, 100, 200)
# cv2.imshow("Final_2", res2)

#Adaptive Threshold
# cv2.adaptiveThreshold(gray, gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 12)
# cv2.imshow("Image_gray", gray)

blur = cv2.GaussianBlur(gray, (9,9), 100)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
close = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel1)
div = np.float32(gray)/(close)
res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
rest, bw_img = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("image_orginal", image)
cv2.imshow("res", res)
cv2.imshow("bw_img", bw_img)

# create new image
h, w = bw_img.shape[:2]
mark = res[0:h, 0:w]

# Bang chu cai
chars = np.array([
    ["A", np.array([1, 0, 0, 0, 0, 0])],
    ["AW", np.array([0, 0, 1, 1, 1, 0])],
    ["AA", np.array([1, 0, 0, 0, 0, 1])],
    ["B", np.array([1, 1, 0, 0, 0, 0])],
    ["C", np.array([1, 0, 0, 1, 0, 0])],
    ["D", np.array([1, 0, 0, 1, 1, 0])],
    ["DD", np.array([0, 1, 1, 1, 0, 1])],
    ["E", np.array([1, 0, 0, 0, 1, 0])],
    ["EE", np.array([1, 1, 0, 0, 0, 1])],
    ["F", np.array([1, 1, 0, 1, 0, 0])],
    ["G", np.array([1, 1, 0, 1, 1, 0])],
    ["H", np.array([1, 1, 0, 0, 1, 0])],
    ["I", np.array([0, 1, 0, 1, 0, 0])],
    ["J", np.array([0, 1, 0, 1, 1, 0])],
    ["K", np.array([1, 0, 1, 0, 0, 0])],
    ["L", np.array([1, 1, 1, 0, 0, 0])],
    ["M", np.array([1, 0, 1, 1, 0, 0])],
    ["N", np.array([1, 0, 1, 1, 1, 0])],
    ["O", np.array([1, 0, 1, 0, 1, 0])],
    ["OO", np.array([1, 0, 0, 1, 1, 1])],
    ["OW", np.array([0, 1, 0, 1, 0, 1])],
    ["P", np.array([1, 1, 1, 1, 0, 0])],
    ["Q", np.array([1, 1, 1, 1, 1, 0])],
    ["R", np.array([1, 1, 1, 0, 1, 0])],
    ["S", np.array([0, 1, 1, 1, 0, 0])],
    ["T", np.array([0, 1, 1, 1, 1, 0])],
    ["U", np.array([1, 0, 1, 0, 0, 1])],
    ["Ư", np.array([1, 1, 0, 0, 1, 1])],
    ["V", np.array([1, 1, 1, 0, 0, 1])],
    ["W", np.array([0, 1, 0, 1, 1, 1])],
    ["X", np.array([1, 0, 1, 1, 0, 1])],
    ["Y", np.array([1, 0, 1, 1, 1, 1])],
    ["Z", np.array([1, 0, 1, 0, 1, 1])]
])

# Ham tim kiem chu cai
def SearchCharacter(array_in):
    con = 0
    for i in range(chars.__len__()):
        if all(array_in == chars[i][1]):
            con = con + 1
            return (chars[i][0])
    if con == 0:
        return "."

# Ham ve khung
def drawRec(imageX, x, y, w, h):
    return cv2.rectangle(imageX, (x, y), (w, h), (0, 255, 0), 1)

#define frame detection local character
w = 15
h = 25
spacing = 8
x = 5
y = 3

check_point = np.array(
    [0, 0, 0, 0, 0, 0]
)

h_x, w_x = gray.shape[:2]

img_1 = Image.new("RGB", (w_x, h_x), (0, 0, 0))
img_3 = np.array(img_1)
img_3 = img_3[:, :, ::-1].copy()
flag = 0
for m in range(0,3):
    for i in range(0,8):
        # Each Character
        drawRec(mark, x, y, x + w, y + h)

        # Moi o nho
        drawRec(mark, x, y, x + w//2, y + h//3) # 1
        drawRec(mark, x, y + h//3, x + w//2, y + (2*h)//3)  # 2
        drawRec(mark, x, y + (2*h)//3, x + w//2, y + h)  # 3
        drawRec(mark, x + w//2, y, x + w, y + h//3) # 4
        drawRec(mark, x + w//2, y + h//3, x + w, y + (2*h)//3) # 5
        drawRec(mark, x + w//2, y + (2*h)//3, x + w, y + h)  # 6
        
        nguong = 10
        im, cnt, hir = cv2.findContours(bw_img[y:y + h//3, x:x + w//2],
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.countNonZero(im) > nguong:
            check_point[0] = 1
        im, cnt, hir = cv2.findContours(bw_img[y + h//3:y + (2*h)//3, x:x + w//2],
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.countNonZero(im) > nguong:
            check_point[1] = 1
        im, cnt, hir = cv2.findContours(bw_img[y + (2*h)//3:y + h, x:x + w//2],
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.countNonZero(im) > nguong:
            check_point[2] = 1
        im, cnt, hir = cv2.findContours(bw_img[y:y + h//3, x + w//2:x + w],
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.countNonZero(im) > nguong:
            check_point[3] = 1
        im, cnt, hir = cv2.findContours(bw_img[y + h//3:y + (2*h)//3, x + w//2:x + w],
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.countNonZero(im) > nguong:
            check_point[4] = 1
        im, cnt, hir = cv2.findContours(bw_img[y + (2*h)//3:y + h, x + w//2:x + w],
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.countNonZero(im) > nguong:
            check_point[5] = 1
        

        # if len(cv2.findContours(bw_img[y:y + h//3, x:x + w//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]) == 1:
        #     check_point[0] = 1
        # if len(cv2.findContours(bw_img[y + h//3:y + (2*h)//3, x:x + w//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]) == 1:
        #     check_point[1] = 1
        # if len(cv2.findContours(bw_img[y + (2*h)//3:y + h, x:x + w//2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]) == 1:
        #     check_point[2] = 1
        # if len(cv2.findContours(bw_img[y:y + h//3, x + w//2:x + w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]) == 1:
        #     check_point[3] = 1
        # if len(cv2.findContours(bw_img[y + h//3:y + (2*h)//3, x + w//2:x + w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]) == 1:
        #     check_point[4] = 1
        # if len(cv2.findContours(bw_img[y + (2*h)//3:y + h, x + w//2:x + w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]) == 1:
        #     check_point[5] = 1
        
        #Custom font
        # img_1 = cv2.imread("braille.jpg")
        # img_1 =cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        # pil_draw = Image.fromarray(img_1)

        # draw = ImageDraw.Draw(pil_draw)
        # font = ImageFont.truetype("SVN-Aptima.ttf", 32)
        # draw.text((x,y+h), "Ư", font=font)

        # img_2 = cv2.cvtColor(np.array(pil_draw), cv2.COLOR_RGB2BGR)
        # cv2.imshow("DRAW", img_2)
        # img_1 = gray[0:h, 0:w]
        # img_1 = Image.fromarray(img_1)

        draw = ImageDraw.Draw(img_1)
        font = ImageFont.truetype("SVN-Aptima.ttf", 32)
        img_2 = draw.text((x, y - 10), SearchCharacter(
            check_point), font=font, fill=(255, 255, 255))

        img_2 = np.array(img_1)
        img_2 = img_2[:,:,::-1].copy()

        img_3 = cv2.add(img_2, img_3)
        cv2.imshow("Text", img_3)

        cv2.putText(mark, SearchCharacter(check_point), (x, y + h + h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        x = x + w + spacing
        check_point = np.array([0, 0, 0, 0, 0, 0])
    flag = flag + 1
    x = 5
    if flag == 1:
        y = 2*h - 3
    elif flag == 2:
        y = 93
    

cv2.imshow("result", cv2.resize(mark, (w_x*2, h_x*2)))
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
