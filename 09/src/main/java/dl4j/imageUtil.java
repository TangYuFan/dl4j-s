package dl4j;


import java.io.*;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;

import javax.swing.*;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.opencv.core.CvType.CV_32S;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR;



/**
*   @desc : 图片工具类
*   @auth : TYF
*   @date : 2019-08-05 - 11:29
*/
public class imageUtil {


    /**
    *   @desc : 图片批量命名
    *   @auth : TYF
    *   @date : 2019-08-05 - 11:29
    */
    public static void reName(String picPath){
        File dir = new File(picPath);
        File out = new File(picPath);
        if(out.exists()){
            out.delete();
        }
        out.mkdir();
        if(dir.isDirectory()){
            File[] images = dir.listFiles();
            //序号1开始
            for(int i=1;i<=images.length;i++){
                if(images[i-1].isFile()){
                    String name = ""+i+images[i-1].getName().substring(images[i-1].getName().indexOf("."));
                    File file = new File(picPath+"/"+name);
                    boolean success = images[i-1].renameTo(file);
                    System.out.println("file:"+file.getAbsolutePath()+", success:"+success);
                }else{
                    System.out.println("this is not file ！");
                }
            }
        }else{
            System.out.println("file path is incorrect！");
        }
    }


    /**
    *   @desc : 图片批量缩放
    *   @auth : TYF
    *   @date : 2019-08-05 - 11:30
    */
    public static void reShape(String picPath,Integer width,Integer height) throws Exception{
        //指定宽高
        File dir = new File(picPath);
        if(dir.isDirectory()){
            File[] images = dir.listFiles();
            for(int i=1;i<=images.length;i++){
                if(images[i-1].isFile()){
                    Mat image = imread(images[i-1].getAbsolutePath(), IMREAD_COLOR);
                    Size size= new Size(width,height);
                    Mat _m = new Mat(size,CV_32S);
                    resize(image,_m,size);
                    imwrite(images[i-1].getAbsolutePath(),_m);
                    System.out.println("file:"+images[i-1].getAbsolutePath());
                }else{
                    System.out.println("this is not file ！");
                }
            }
        }else{
            System.out.println("file path is incorrect！");
        }
    }


    /**
    *   @desc : 图片批量截取
    *   @auth : TYF
    *   @date : 2019-08-05 - 11:33
    */
    public static void reCutOut(String picPath,Integer width,Integer height,Integer x,Integer y){
        //左上角坐标点xy
        File dir = new File(picPath);
        if(dir.isDirectory()){
            File[] images = dir.listFiles();
            for(int i=1;i<=images.length;i++){
                if(images[i-1].isFile()){
                    try{
                        Mat image = imread(images[i-1].getAbsolutePath(), IMREAD_COLOR);
                        Mat rect = new Mat(image,new Rect(x,y,width,height));
                        imwrite(images[i-1].getAbsolutePath(),rect);
                    }catch (Exception e){
                        //截取框超出原图
                        continue;
                    }
                }else{
                    System.out.println("this is not file ！");
                }
            }
        }else{
            System.out.println("file path is incorrect！");
        }
    }


    public static void main(String[] args) throws Exception{

        //拉伸为608x608
        reShape("D:\\my_idea_workspace\\dl4j\\deeplearning4j\\09\\JPEGImages",960,540);
        //重命名递增序号
        reName("D:\\my_idea_workspace\\dl4j\\deeplearning4j\\09\\JPEGImages");
        //原图截取608x608
        reCutOut("D:\\my_idea_workspace\\dl4j\\deeplearning4j\\09\\JPEGImages",960,540,0,0);

    }


}
