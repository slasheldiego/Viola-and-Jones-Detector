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
DIR *dir2;
DIR *dir3;
struct dirent *ent;
struct dirent *ent2;
struct dirent *ent3;
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
//GpuMat eyesBuf_gpu;
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
    char path_src_sub[100];
    char mes[2];
    char dia[2];
    strcpy( mes, argv[1] );
    strcpy( dia, argv[2] );

	
    printf( "%s\n", dia );
 
    strcpy( path, "/home/diego/Imágenes/CK+/cohn-kanade-images/" );
    //strcpy( path, "/home/diego/Imágenes/BIOid/" );
    //sprintf( path_src, "/home/diego/Imágenes/colorferet/colorferet/colorferet/dvd%s/data/images/%s/", mes, dia );
    //strcpy( path_src, path );
    //std::cout << file_img << numstr;
    if ((dir = opendir (path)) != NULL) {
        /* imprimo todo los archivos que estan en la carpeta de imagenes de personas detectadas */
        while ((ent = readdir (dir)) != NULL) {
            //file = ent->d_name;
            //printf("===>%s\n",ent->d_name);
            strcpy( path_src, path );
	    strcat( path_src, ent->d_name );
            strcat( path_src, "/" );
            //printf( "waaaa %s\n", path_src );
            if( ( dir2 = opendir(path_src) ) != NULL && strcmp( ent->d_name, ".") != 0 && strcmp( ent->d_name, "..") != 0 ){
               while( ( ent2 = readdir(dir2) ) != NULL ){
                  printf( "====>%s\n", ent2->d_name );
                  strcpy( path_src_sub, path_src ); 
                  strcat( path_src_sub, ent2->d_name ); 
                  strcat( path_src_sub, "/" );               
 
                  if( ( dir3 = opendir( path_src_sub ) ) != NULL && strcmp( ent2->d_name, ".") != 0 && strcmp( ent2->d_name, "..") != 0 ){
                        while( ( ent3 = readdir(dir3) ) != NULL ){
                        file = ent3->d_name;

                  	int pos = (int) file.find(".png");
            		if (pos != -1){
                		std::cout << "waaaaa " << path_src_sub + file <<std::endl;
                		//num_img++;//PARA SISTEMA PERSISTENTE

                		image = cv::imread(path_src_sub+file);//+file_img);

               			//imshow("test",image);
                		//waitKey();
                		cv::CascadeClassifier eyeCascade;
                		//cv::gpu::CascadeClassifier_GPU eyeCascade;
 
                		if (!eyeCascade.load(xml)) {
                    		std::cout << "Erro carregando faceCascade. Funcao \"detect_eye\"." << std::endl;
                    		return -1;
                		}
                
                		cvtColor(image, frame_gray, COLOR_BGR2GRAY);
                		equalizeHist(frame_gray, frame_gray);
                		//GpuMat gray_gpu(frame_gray);
                
                		std::vector<cv::Rect> eyes;
                
                		//En el paper utilizan el parametro de scala 1.05 aqui es de 1.1
                		//faceCascade.detectMultiScale(frame_gray, faces, 1.01, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(30,30));
                		//eyeCascade.visualizeInPlace = false;
                		//eyeCascade.findLargestObject = true;
                		eyeCascade.detectMultiScale(frame_gray, eyes, 1.05, 3, 0 | CASCADE_SCALE_IMAGE, cv::Size(3,3));
                		//eyeCascade.detectMultiScale( gray_gpu, eyesBuf_gpu, 1.05, 4, cv::Size(3,3));
                
                		// Set Region of Interest
                		cv::Rect roi_b;
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
                        			imwrite( path+std::string("eye-deteccion/eye-detection-")+ent->d_name+std::string("-")+numstr2+std::string("-")+file, crop );
                        			//imwrite( path+std::string("BIOid-eyeDetection/")+numstr2+std::string("_")+file, crop );
                        			cv::imshow("detected", crop);
                        			num_obj++;
                    			}
                    			else
                        			destroyWindow("detected");
                    
                    
                		}
                
                // Show image
                //sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
                 //text = sstm.str();
                
                //putText(image, "Imagen de prueba", cv::Point(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 255), 1, 0);
                //cv::imshow("original", image);
                //waitKey(0);
            		    }
                        }
                        closedir( dir3 );
                    }
               }
               closedir( dir2 );
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

