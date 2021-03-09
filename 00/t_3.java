import org.bytedeco.javacv.CanvasFrame;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * @desc : 卷积算法实现
 * @auth : TYF
 * @date : 2019-06-21 - 10:51
 */
public class t_3 {

    /**
     * 灰度图卷积
     *
     * @param input        2维图片
     * @param x            卷积位置x坐标
     * @param y            卷积位置y坐标
     * @param k            2维卷积内核
     * @param kernelWidth  卷积核width
     * @param kernelHeight 卷积核height
     * @return 卷积后的2维图片
     */
    public static double singlePixelConvolution(double[][] input,
                                                int x, int y,
                                                double[][] k,
                                                int kernelWidth,
                                                int kernelHeight) {
        double output = 0;
        for (int i = 0; i < kernelWidth; ++i) {
            for (int j = 0; j < kernelHeight; ++j) {
                output = output + (input[x + i][y + j] * k[i][j]);
            }
        }
        return output;
    }

    /**
     * 灰度图卷积(指定宽高)
     *
     * @param input        2维图片
     * @param width        图片width
     * @param height       图片height
     * @param kernel       2维卷积内核
     * @param kernelWidth  卷积核width
     * @param kernelHeight 卷积核height
     * @return 卷积后的2维图片
     */
    public static double[][] convolution2D(double[][] input,
                                           int width, int height,
                                           double[][] kernel,
                                           int kernelWidth,
                                           int kernelHeight) {
        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        double[][] output = new double[smallWidth][smallHeight];
        for (int i = 0; i < smallWidth; ++i) {
            for (int j = 0; j < smallHeight; ++j) {
                output[i][j] = 0;
            }
        }
        for (int i = 0; i < smallWidth; ++i) {
            for (int j = 0; j < smallHeight; ++j) {
                output[i][j] = singlePixelConvolution(input, i, j, kernel,
                        kernelWidth, kernelHeight);
            }
        }
        return output;
    }

    /**
     * 灰度图卷积(指定宽高、区域)
     *
     * @param input        2维图片
     * @param width        图片width
     * @param height       图片height
     * @param kernel       2维卷积内核
     * @param kernelWidth  卷积核width
     * @param kernelHeight 卷积核height
     * @return 卷积后的2维图片
     */
    public static double[][] convolution2DPadded(double[][] input,
                                                 int width, int height,
                                                 double[][] kernel,
                                                 int kernelWidth,
                                                 int kernelHeight) {
        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        int top = kernelHeight / 2;
        int left = kernelWidth / 2;

        double[][] small = convolution2D(input, width, height,
                kernel, kernelWidth, kernelHeight);
        double large[][] = new double[width][height];
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                large[i][j] = 0;
            }
        }
        for (int j = 0; j < smallHeight; ++j) {
            for (int i = 0; i < smallWidth; ++i) {
                large[i + left][j + top] = small[i][j];
            }
        }
        return large;
    }

    /**
     * 灰度图多次卷积(指定宽高、区域)
     *
     * @param input        2维图片
     * @param width        图片width
     * @param height       图片height
     * @param kernel       2维卷积内核
     * @param kernelWidth  卷积核width
     * @param kernelHeight 卷积核height
     * @param iterations   卷积次数
     * @return 卷积后的2维图片
     */
    public double[][] convolutionType2(double[][] input,
                                       int width, int height,
                                       double[][] kernel,
                                       int kernelWidth, int kernelHeight,
                                       int iterations) {
        double[][] newInput = input.clone();
        double[][] output = input.clone();

        for (int i = 0; i < iterations; ++i) {
            output = convolution2DPadded(newInput, width, height,
                    kernel, kernelWidth, kernelHeight);
            newInput = output.clone();
        }
        return output;
    }

    //INDArray转2维数组
    public static double[][] indArray2Array(INDArray array){
        long[] shape = array.shape();
        long height = shape[2];
        long width = shape[3];
        double[][] out = new double[(int)height][(int)width];
        BufferedImage image = new BufferedImage((int)width, (int)height, BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double value = array.getDouble(0, 0, y, x);
                out[x][y]=value;
            }
        }
        return out;
    }

    //2维数组转图片
    public static BufferedImage array2Image(double[][] in,int h,int w){
        BufferedImage image = new BufferedImage(w,h, BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                double red = in[x][y];
                double green = in[x][y];
                double blue = in[x][y];
                //handle out of bounds pixel values
                red = Math.min(red, 255);
                green = Math.min(green, 255);
                blue = Math.min(blue, 255);
                red = Math.max(red, 0);
                green = Math.max(green, 0);
                blue = Math.max(blue, 0);
                image.setRGB(x, y, new Color((int)red, (int)green, (int)blue).getRGB());
            }
        }
        return image;
    }


    public static void test_1() throws Exception{
        //加载图片矩阵
        String file = "D:/my_dl4j/car_number_detection/pics/1.jpg";
        NativeImageLoader LOADER = new NativeImageLoader(512, 512, 1);
        INDArray content = LOADER.asMatrix(new File(file));
        //获取数组
        double[][] in = indArray2Array(content);
        //二维数组转图片
        BufferedImage out = array2Image(in,512, 512);
        //显示图片
        CanvasFrame canvas = new CanvasFrame("pic", 1);
        canvas.showImage(out);
    }


    public static void test_2()throws Exception{
        //加载图片矩阵
        String file = "D:/my_dl4j/car_number_detection/pics/1.jpg";
        NativeImageLoader LOADER = new NativeImageLoader(512, 512, 1);
        INDArray content = LOADER.asMatrix(new File(file));
        //获取数组
        double[][] in = indArray2Array(content);
        //------------------------------测试-------------------------------------
        //创建卷积核
        double[][] k_vertical = new double[][]{ new double[]{1,0,-1},new double[]{1,0,-1},new double[]{1,0,-1}};
        double[][] k_sobel_vertical = new double[][]{ new double[]{1,0,-1},new double[]{2,0,-2},new double[]{1,0,-1}};
        double[][] k_scharr_vertical = new double[][]{ new double[]{3,0,-3},new double[]{10,0,-10},new double[]{3,0,-3}};
        double[][] k_horizontal = new double[][]{ new double[]{1,1,1},new double[]{0,0,0},new double[]{-1,-1,-1}};
        double[][] k_sobel_horizontal = new double[][]{ new double[]{1,2,1},new double[]{0,0,0},new double[]{-1,2,-1}};
        double[][] k_scharr_horizontal = new double[][]{ new double[]{3,10,3},new double[]{0,0,0},new double[]{-3,10,-3}};
        double[][] out_1 = convolution2DPadded(in,512,512,k_vertical,3,3);
        double[][] out_2 = convolution2DPadded(in,512,512,k_sobel_vertical,3,3);
        double[][] out_3 = convolution2DPadded(in,512,512,k_scharr_vertical,3,3);
        double[][] out_4 = convolution2DPadded(in,512,512,k_horizontal,3,3);
        double[][] out_5 = convolution2DPadded(in,512,512,k_sobel_horizontal,3,3);
        double[][] out_6 = convolution2DPadded(in,512,512,k_scharr_horizontal,3,3);
        //------------------------------测试-------------------------------------
        //二维数组转图片
        BufferedImage image_1 = array2Image(out_1,512, 512);
        BufferedImage image_2 = array2Image(out_2,512, 512);
        BufferedImage image_3 = array2Image(out_3,512, 512);
        BufferedImage image_4 = array2Image(out_4,512, 512);
        BufferedImage image_5 = array2Image(out_5,512, 512);
        BufferedImage image_6 = array2Image(out_6,512, 512);
        //显示图片
        new CanvasFrame("k_vertical", 1).showImage(image_1);
        new CanvasFrame("k_sobel_vertical", 1).showImage(image_2);
        new CanvasFrame("k_scharr_vertical", 1).showImage(image_3);
        new CanvasFrame("k_horizontal", 1).showImage(image_4);
        new CanvasFrame("k_sobel_horizontal", 1).showImage(image_5);
        new CanvasFrame("k_scharr_horizontal", 1).showImage(image_6);

    }

    public static void main(String[] args) throws Exception{
        test_2();
    }



}