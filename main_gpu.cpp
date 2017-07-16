//
//  main.cpp
//  DeteccionOjos
//
//  Created by Diego Benavides on 14/07/15.
//  Copyright (c) 2015 Diego Benavides. All rights reserved.
//

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <iostream>
#include <stdio.h>
#include <string.h>
//#include <CoreFoundation/CoreFoundation.h>
#include <cassert>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/core/opengl_interop.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace cv;
using namespace cv::gpu;
using namespace std;

//Variables de inspector de imagenes
DIR *dir;
struct dirent *ent;
std::string file_img;
std::string file;
int num_img = 0;
char numstr[21];
int num_obj = 1;
char numstr2[3];
std::string auxstring;
int cont = 14;


//Variables de detector
//const std::string xml = "/Users/diegobenavides/Documents/Cursos de mestrado/Fundamentos de sistemas inteligentes/Trabalho Final/codigos/detector-p-criminal/detector-p-criminal/haarcascade_frontalface_alt.xml";
const std::string xml = "haarcascade_eye.xml";

cv::Mat image; //= cv::imread("/Users/diegobenavides/Documents/Cursos de mestrado/Fundamentos de sistemas inteligentes/Trabalho Final/codigos/opencv-test/opencv-test/test9.jpg");
cv::Mat frame_gray;
cv::Mat crop;
cv::Mat res;
cv::Mat gray;
string text;
stringstream sstm;
GpuMat eyes;
//GpuMat gray_gpu;

int main(int argc, const char * argv[]) {
    
    
    //persistencia del inspector y detector de rostros
    //while (true)
    //{
    num_img++;
    //construyo el nombre del archivo a buscar
    //sprintf(numstr, "%d", num_img);
    //file_img = std::string("image-persona") + numstr + std::string(".jpg");
    char path[100];
    char path_src[100];
    char mes[2];
    char dia[2];
    //strcpy( mes, argv[1] );
    //strcpy( dia, argv[2] );
    int detections_num;
	
    //printf( "%s\n", dia );
 
    //sprintf( path, "/home/diego/Imágenes/dataset-images/2002/%s/%s/", mes, dia );
    strcpy( path, "/home/diego/Imágenes/BIOid/" );
    //sprintf( path_src, "/home/diego/Imágenes/dataset-images/2002/%s/%s/big/", mes, dia );
    strcpy( path_src, "/home/diego/Imágenes/BIOid/BIOid-FaseDataBase/" );

    //std::cout << file_img << numstr;
    if ((dir = opendir (path_src)) != NULL) {
        /* imprimo todo los archivos que estan en la carpeta de imagenes de personas detectadas */
        while ((ent = readdir (dir)) != NULL) {
            file = ent->d_name;
            
            //Confirmo la existencia de la imagen actual
            //if(file.compare(file_img) == 0)
            int pos = (int) file.find(".pgm");
            if (pos != -1)
            {
                //std::cout << "waaaaa " << file <<std::endl;
                //num_img++;//PARA SISTEMA PERSISTENTE
                
                image = cv::imread(path_src+file);//+file_img);
                
                //cv::CascadeClassifier eyeCascade;
                cv::gpu::CascadeClassifier_GPU eyeCascade;
 
                if (!eyeCascade.load(xml)) {
                    std::cout << "Erro carregando faceCascade. Funcao \"detect_eye\"." << std::endl;
                    return -1;
                }
                
                cvtColor(image, frame_gray, COLOR_BGR2GRAY);
                equalizeHist(frame_gray, frame_gray);
                GpuMat gray_gpu(frame_gray);
                Mat eyes_downloaded;
                //std::vector<cv::Rect> eyes;
                std::vector<cv::Rect> eyes_res;
                
                //En el paper utilizan el parametro de scala 1.05 aqui es de 1.1
                //faceCascade.detectMultiScale(frame_gray, faces, 1.01, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(30,30));
                eyeCascade.visualizeInPlace = false;
                eyeCascade.findLargestObject = true;
                //eyeCascade.detectMultiScale(frame_gray, eyes, 1.05, 3, 0 | CASCADE_SCALE_IMAGE, cv::Size(3,3));
                detections_num = eyeCascade.detectMultiScale( gray_gpu, eyes, 1.05, 4, cv::Size(3,3));
                
                if(detections_num!=0){

                   eyes.colRange(0, detections_num).download(eyes_downloaded);
                   Rect *eyeRects = eyes_downloaded.ptr<Rect>();

                   for(int i=0;i<detections_num;i++){
                      eyes_res.push_back(eyeRects[i]);
                   }

		}
                gray_gpu.release();
                eyes.release();

                
                // Set Region of Interest
                /*cv::Rect roi_b;
                cv::Rect roi_c;
                
                size_t ic = 0; // ic is index of current element
                int ac = 0; // ac is area of current element
                
                size_t ib = 0; // ib is index of biggest element
                int ab = 0; // ab is area of biggest element
                
                num_obj = 1;
                if(eyes.size() == 0){
		   std::cout << "No se detectaron ojos en: " << file <<std::endl;
                   //imwrite( std::string("/Users/diegobenavides/Documents/DataSets/OjosDataSet/2002/08/13/eye-detection-")+file, image);
                }

                for (ic = 0; ic < eyes.size(); ic++) // Iterate through all current elements (detected faces)
                    
                {
                    roi_c.x = eyes[ic].x;
                    roi_c.y = eyes[ic].y;
                    roi_c.width = (eyes[ic].width);
                    roi_c.height = (eyes[ic].height);
                    
                    ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)
                    
                    roi_b.x = eyes[ib].x;
                    roi_b.y = eyes[ib].y;
                    roi_b.width = (eyes[ib].width);
                    roi_b.height = (eyes[ib].height);
                    
                    ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element
                    
                    if (ac > ab)
                    {
                        ib = ic;
                        roi_b.x = eyes[ib].x;
                        roi_b.y = eyes[ib].y;
                        roi_b.width = (eyes[ib].width);
                        roi_b.height = (eyes[ib].height);
                    }
                    
                    crop = image(roi_b);
                    //resize(crop, res, cv::Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
                    cvtColor(crop, gray, cv::COLOR_BGR2GRAY); // Convert cropped image to Grayscale
                    
                    
                    //cv::Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
                    //cv::Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
                    //rectangle(image, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
                    
                    if (!crop.empty())
                    {
                        sprintf(numstr2, "%d", num_obj);
                        std::cout << "Deteccion ..." << std::endl;
                        //imwrite( std::string("/Users/diegobenavides/Documents/Cursos de mestrado/Fundamentos de sistemas inteligentes/Trabalho Final/codigos/detector-p-criminal/detector-p-criminal/image-detection-face/face-detection")+numstr+std::string("-")+numstr2+std::string(".jpg"), crop );
                        //imwrite( std::string("/Users/diegobenavides/Documents/Cursos de mestrado/Fundamentos de sistemas inteligentes/Trabalho Final/codigos/detector-p-criminal/detector-p-criminal/image-detection-face/Detectados-face/face-detection-")+file+std::string("-")+numstr2+std::string(".jpg"), crop );
                        imwrite( path+std::string("eye-deteccion/eye-detection-")+argv[1]+std::string("-")+argv[2]+std::string("-")+numstr2+std::string("-")+file, crop );
                        cv::imshow("detected", crop);
                        num_obj++;
                        //waitKey(0);
                    }
                    else
                        destroyWindow("detected");
                    
                    
                }
                
                // Show image
                sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
                 text = sstm.str();*/
                
                //putText(image, "Imagen de prueba", cv::Point(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 255), 1, 0);
                //cv::imshow("original", image);
                //waitKey(0);
            }
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }
    
    //}
    
    return 0;
}

