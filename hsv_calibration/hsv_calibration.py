import requests
import cv2
import numpy as np
import argparse

parser=argparse.ArgumentParser(description="Real-Time range for HSV Values")
group=parser.add_mutually_exclusive_group()
group.add_argument("-i","--image",type=str, metavar="", help="Image Mode, Pass URL: Example: xyz.py -i Downloads/image.jpg",dest="image")
group.add_argument("-c","--camera",type=str, metavar="", help="Camera Mode, Pass IP Address: Example: xyz.py -c http://192.168.2.87:8081/",dest="camera")
group.add_argument("-w","--webcam", action="store_true", help="Webcam Mode, Pass NOTHING: Example: xyz.py -w",dest="webcam")
group.add_argument("-s","--size", type=int, metavar="", help="(Must be last Parameter) Resize Image: Example: xyz.py -i Downloads/image.jpg -s 800 600",dest="x")


args, unknown = parser.parse_known_args()

class TrackBars:
    name="Input"   
    
    def __init__(self):
        def val(x):
            pass
        name=self.name
        cv2.namedWindow(name)
        cv2.resizeWindow(name,600,800)
        cv2.createTrackbar("Brightness",name,0,99,val)
        cv2.createTrackbar("Darkness",name,0,99,val)
        cv2.createTrackbar("Normal Blur",name,0,99,val)
        cv2.createTrackbar("Gaussian Blur",name,0,99,val)
        cv2.createTrackbar("Gaussian Deviation",name,0,99,val)
        cv2.createTrackbar("Median Blur",name,0,99,val)
        cv2.createTrackbar("Bilateral Blur",name,0,11,val)
        cv2.createTrackbar("Sigma Color",name,0,999,val)
        cv2.createTrackbar("Sigma Space",name,0,999,val)
        cv2.createTrackbar("H Low",name,0,179,val)
        cv2.createTrackbar("H High",name,0,179,val)
        cv2.createTrackbar("S Low",name,0,255, val)
        cv2.createTrackbar("S High",name,0,255, val)
        cv2.createTrackbar("V Low",name,0,255, val)
        cv2.createTrackbar("V High",name,0,255, val)
        cv2.setTrackbarPos("H High",name,255)
        cv2.setTrackbarPos("S High",name,255)
        cv2.setTrackbarPos("V High",name,255)
        
        cv2.createTrackbar("Switch 1: Erosion-Dilaton 2:Dilation-Erosion",name,0,2,val)
        cv2.createTrackbar("Erosion",name,0,25, val)
        cv2.createTrackbar("Erosion Iterations",name,1,20, val)
        cv2.createTrackbar("Dilation",name,0,25, val)
        cv2.createTrackbar("Dilation Iterations",name,1,20, val)
        
        cv2.createTrackbar("Canny: Threshold 1",name,0,1000, val)
        cv2.createTrackbar("Canny: Threshold 2",name,0,1000, val)
        
    
    
    def get_all_values(self):
        name=self.name
        empty_img=np.zeros((700,1000,3),dtype="uint8")
        x=0
        y=0
        
        brightness=cv2.getTrackbarPos("Brightness",name)
        darkness=cv2.getTrackbarPos("Darkness",name)
        b_blur=cv2.getTrackbarPos("Normal Blur",name)
        g_blur=cv2.getTrackbarPos("Gaussian Blur",name)
        g_dev=cv2.getTrackbarPos("Gaussian Deviation",name)
        m_blur=cv2.getTrackbarPos("Median Blur",name)
        bl_blur=cv2.getTrackbarPos("Bilateral Blur",name)
        sigma_color=cv2.getTrackbarPos("Sigma Color",name)
        sigma_space=cv2.getTrackbarPos("Sigma Space",name)
        h_low=cv2.getTrackbarPos("H Low",name)
        h_high=cv2.getTrackbarPos("H High",name)
        s_low=cv2.getTrackbarPos("S Low",name)
        s_high=cv2.getTrackbarPos("S High",name)
        v_low=cv2.getTrackbarPos("V Low",name)
        v_high=cv2.getTrackbarPos("V High",name)
        
        switch_val=cv2.getTrackbarPos("Switch 1: Erosion-Dilaton 2:Dilation-Erosion",name)
        erosion=cv2.getTrackbarPos("Erosion",name)
        erosion_iters=cv2.getTrackbarPos("Erosion Iterations",name)
        dilation=cv2.getTrackbarPos("Dilation",name)
        dilation_iters=cv2.getTrackbarPos("Dilation Iterations",name)
        
        canny_t1=cv2.getTrackbarPos("Canny: Threshold 1",name)
        canny_t2=cv2.getTrackbarPos("Canny: Threshold 2",name)
        
        if((b_blur%2==0) & (b_blur!=0)):
            b_blur=b_blur+1
        if((g_blur%2==0) & (g_blur!=0)):
            g_blur=g_blur+1
        if((m_blur%2==0) & (m_blur!=0)):
            m_blur=m_blur+1
        if((bl_blur%2==0) & (bl_blur!=0)):
            bl_blur=bl_blur+1
        
        variables_names={   "Brightness":brightness,
                            "Darkness":darkness,                            
                            "Box Blur Kernel":b_blur,
                            "Gaussian Blur Kernel":g_blur,
                            "Gaussian Deviation":g_dev,
                            "Median Blur Kernel":m_blur,
                            "Bilateral Blur Kernel":bl_blur,
                            "Bilateral Sigma Color":sigma_color,
                            "Bilateral Sigma Space":sigma_space,
                            "Hue Low":h_low,
                            "Hue High":h_high,
                            "Saturation Low":s_low,
                            "Saturation High":s_high,
                            "Value Low":v_low,
                            "Value High":v_high,
                            "Switch (0: None 1: Erosion then Dilation 2: Dilation then Erosion):":switch_val,
                            "Erosion Kernel":erosion,
                            "Erosion Iterations":erosion_iters,
                            "Dilation Kernel":dilation,
                            "Dilation Iterations":dilation_iters,
                            "Canny Threshold 1":canny_t1,
                            "Canny Threshold 2":canny_t2}
        
        
        for idx,(key,value) in enumerate(variables_names.items(),start=1):
            cv2.putText(empty_img,str(key)+" = "+str(value),(x,y+idx*25),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        
        cv2.putText(empty_img,"HSV Lower Range: [ "+str(h_low)+", "+str(s_low)+", "+str(v_low)+" ]",(x,y+590),cv2.FONT_ITALIC,1,(0,255,0))
        cv2.putText(empty_img,"HSV Upper Range: [ "+str(h_high)+", "+str(s_high)+", "+str(v_high)+" ]",(x,y+630),cv2.FONT_ITALIC,1,(0,255,0))
    
        return empty_img
        
    def get_values(self):
        name=self.name
        hue_low=cv2.getTrackbarPos("H Low",name)
        hue_high=cv2.getTrackbarPos("H High",name)
        sat_low=cv2.getTrackbarPos("S Low",name)
        sat_high=cv2.getTrackbarPos("S High",name)
        val_low=cv2.getTrackbarPos("V Low",name)
        val_high=cv2.getTrackbarPos("V High",name)
        
        if(hue_high<=hue_low):
            cv2.setTrackbarPos("H High",name,hue_low)
            
        if(hue_high<=hue_low):
            cv2.setTrackbarPos("H Low",name,hue_high)
            
        if(sat_high<=sat_low):
            cv2.setTrackbarPos("S High",name,sat_low)
            
        if(sat_high<=sat_low):
            cv2.setTrackbarPos("S Low",name,sat_high)
        
        if(val_high<=val_low):
            cv2.setTrackbarPos("V High",name,val_low)
        
        if(val_high<=val_low):
            cv2.setTrackbarPos("V Low",name,val_high)  
        
        return np.array([hue_low,sat_low,val_low]),np.array([hue_high,sat_high,val_high])
    
    def blur_image(self,image):
        b_blur=cv2.getTrackbarPos("Normal Blur",self.name)
        g_blur=cv2.getTrackbarPos("Gaussian Blur",self.name)
        m_blur=cv2.getTrackbarPos("Median Blur",self.name)
        bl_blur=cv2.getTrackbarPos("Bilateral Blur",self.name)
        
        g_deviation=cv2.getTrackbarPos("Guassian Deviation",self.name)
        sigma_color=cv2.getTrackbarPos("Sigma Color",self.name)
        sigma_space=cv2.getTrackbarPos("Sigma Space",self.name)
        
        if(b_blur>0):
            cv2.setTrackbarPos("Gaussian Blur",self.name,0)
            cv2.setTrackbarPos("Median Blur",self.name,0)
            cv2.setTrackbarPos("Bilateral Blur",self.name,0)
            cv2.setTrackbarPos("Guassian Deviation",self.name,0)
            cv2.setTrackbarPos("Sigma Color",self.name,0)
            cv2.setTrackbarPos("Sigma Space",self.name,0)
            
            blur_val=b_blur
            if((blur_val%2)==1):
                blurred_img=cv2.blur(image,(blur_val,blur_val))
            else:
                blurred_img=cv2.blur(image,(blur_val+1,blur_val+1))
            return blurred_img
        
        if(g_blur>0):
            cv2.setTrackbarPos("Normal Blur",self.name,0)
            cv2.setTrackbarPos("Median Blur",self.name,0)
            cv2.setTrackbarPos("Bilateral Blur",self.name,0)
            cv2.setTrackbarPos("Sigma Color",self.name,0)
            cv2.setTrackbarPos("Sigma Space",self.name,0)
            blur_val=g_blur
            if((blur_val%2)==1):
                blurred_img=cv2.GaussianBlur(image,(blur_val,blur_val),g_deviation)
            else:
                blurred_img=cv2.GaussianBlur(image,(blur_val+1,blur_val+1),g_deviation)
            return blurred_img
        
        if(m_blur>0):
            cv2.setTrackbarPos("Gaussian Blur",self.name,0)
            cv2.setTrackbarPos("Normal Blur",self.name,0)
            cv2.setTrackbarPos("Bilateral Blur",self.name,0)
            cv2.setTrackbarPos("Guassian Deviation",self.name,0)
            cv2.setTrackbarPos("Sigma Color",self.name,0)
            cv2.setTrackbarPos("Sigma Space",self.name,0)
            blur_val=m_blur
            if((blur_val%2)==1):
                blurred_img=cv2.medianBlur(image,blur_val)
            else:
                blurred_img=cv2.medianBlur(image,blur_val+1)
            return blurred_img
        
        if(bl_blur>0):
            cv2.setTrackbarPos("Gaussian Blur",self.name,0)
            cv2.setTrackbarPos("Median Blur",self.name,0)
            cv2.setTrackbarPos("Normal Blur",self.name,0)
            cv2.setTrackbarPos("Guassian Deviation",self.name,0)
        
            blur_val=bl_blur
            if((blur_val%2)==1):
                blurred_img=cv2.bilateralFilter(image,blur_val,sigma_color,sigma_space)
            else:
                blurred_img=cv2.bilateralFilter(image,blur_val+1,sigma_color,sigma_space)
            return blurred_img
        
        if(((b_blur==0) & (g_blur==0)) & ((bl_blur==0) & (m_blur==0))):
            blurred_img=cv2.blur(image,(1,1))
            return blurred_img


    def image_edges(self,image):
        t1=cv2.getTrackbarPos("Canny: Threshold 1",self.name)
        t2=cv2.getTrackbarPos("Canny: Threshold 2",self.name)
        canny_image=cv2.Canny(image,t1,t2)
        return canny_image
    
    def erosion_image(self,image):
        erosion_val=cv2.getTrackbarPos("Erosion",self.name)
        erosion_kernel=np.ones((erosion_val,erosion_val),np.uint8)
        erosion_iterations=cv2.getTrackbarPos("Erosion Iterations",self.name)
        erosion_image=cv2.erode(image,erosion_kernel,iterations=erosion_iterations)
        return erosion_image
    
    def dilation_image(self,image):
        dilate_val=cv2.getTrackbarPos("Dilation",self.name)
        dilate_kernel=np.ones((dilate_val,dilate_val),np.uint8)
        dilate_iterations=cv2.getTrackbarPos("Dilation Iterations",self.name)
        dilate_image=cv2.dilate(image,dilate_kernel,iterations=dilate_iterations)
        return dilate_image
    
    
    def switch_e_d(self):
        switch_val=cv2.getTrackbarPos("Switch 1: Erosion-Dilaton 2:Dilation-Erosion",self.name)
        return switch_val
    
    def brightness_darkness(self,image):
        brightness=cv2.getTrackbarPos("Brightness",self.name)
        darkness=cv2.getTrackbarPos("Darkness",self.name)
        
        b_kernel=np.ones(image.shape,dtype="uint8")*brightness
        d_kernel=np.ones(image.shape,dtype="uint8")*darkness
        
        bright_img=cv2.add(image,b_kernel)
        dark_img=cv2.subtract(bright_img,d_kernel)
        
        return dark_img
        
        
        
        
        
class FromImage:
    def __init__(self,url):
        self.image=cv2.imread(url)
        
    def read_image(self):
        return self.image
    
        
class FromWebcam:
    def __init__(self):
        self.webcam=cv2.VideoCapture(0)
          
    def read_image(self):
        _,self.frame=self.webcam.read(0)
        self.frame = cv2.flip(self.frame,1)
        return self.frame 
    
    def webcam_release(self):
        self.webcam.release()
        
        

class ProcessImage:  
    
    
    
    def inner_process(self,init_img,trackbars):
        
        bright_dark_img=trackbars.brightness_darkness(init_img)
        blurred_img=trackbars.blur_image(bright_dark_img)
        
        hsv_img=cv2.cvtColor(blurred_img,cv2.COLOR_BGR2HSV)
        lower_range,upper_range=trackbars.get_values()

        if(trackbars.switch_e_d()==1):
            eroded_img=trackbars.erosion_image(hsv_img)
            dilated_img=trackbars.dilation_image(eroded_img)
            mask=cv2.inRange(dilated_img,lower_range,upper_range)
        
        elif(trackbars.switch_e_d()==2):
            dilated_img=trackbars.dilation_image(hsv_img)
            eroded_img=trackbars.erosion_image(dilated_img)
            mask=cv2.inRange(eroded_img,lower_range,upper_range)
        else:
            mask=cv2.inRange(hsv_img,lower_range,upper_range)
        
        
        edge_mask_img=trackbars.image_edges(mask)
        edge_blurred_img=trackbars.image_edges(blurred_img)
        edge_hsv_img=trackbars.image_edges(hsv_img)
        output=cv2.bitwise_and(init_img,init_img,mask=mask)
        

        output_vars=trackbars.get_all_values()
        
        cv2.imshow("Values Output",output_vars)
        cv2.imshow("Input Image",init_img)
        cv2.imshow("Blurred", blurred_img)
        cv2.imshow("Mask", mask)
        cv2.imshow("Edge Detection on MASK", edge_mask_img)
        cv2.imshow("Edge Detection on HSV Converted Image", edge_hsv_img)
        cv2.imshow("Edge Detection on BLURRED IMAGE", edge_blurred_img)
        cv2.imshow("Output",output)

    @staticmethod
    def process_webcam():
        try:
            trackbars=TrackBars()
            cls_image=FromWebcam()
            while True:
                init_img=cls_image.read_image()
                ProcessImage().inner_process(init_img,trackbars)
                if(cv2.waitKey(1)==13):
                    break
        finally:
            cls_image.webcam_release()
            cv2.destroyAllWindows()
            
        
    @staticmethod
    def process_image(url):
        try:
            trackbars=TrackBars()
            cls_image=FromImage(url)
            while True:
                init_img=cls_image.read_image()
                ProcessImage().inner_process(init_img,trackbars)
                if(cv2.waitKey(1)==13):
                    break
        finally:
            cv2.destroyAllWindows()
            
    @staticmethod
    def process_mobile_camera(url):
        try:
            trackbars=TrackBars()
            url=url+"shot.jpg"
            while True:
                imgRequest=requests.get(url)
                imgArray=np.array(bytearray(imgRequest.content),dtype=np.uint8)
                init_img=cv2.imdecode(imgArray,-1)
                ProcessImage().inner_process(init_img,trackbars)
                if(cv2.waitKey(1)==13):
                    break
        finally:
            cv2.destroyAllWindows()
        
        
def main(name,url=None): 
    if(name=="image"):
        ProcessImage().process_image(url)
    elif(name=="webcam"):
        ProcessImage().process_webcam()
    elif(name=="camera"):
        ProcessImage().process_mobile_camera(url)
        
        
    
if __name__=="__main__":
    if not (args.image or args.webcam or args.camera):
        parser.error("Please pass atleast one argument. --h for help")
    elif(args.image):
        main("image",args.image)
    elif(args.webcam):
        main("webcam")
    elif(args.camera):
        main("camera",args.camera)