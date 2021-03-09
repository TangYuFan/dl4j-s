package dl4j;

import org.bytedeco.javacv.CanvasFrame;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Map;

/**
 * @desc : 获取中间层输出
 * @auth : TYF
 * @date : 2019-06-18 - 13:22
 */
public class vggImageNetCut {

    //加载样本
    public static INDArray loadPic(String path) throws Exception{
        NativeImageLoader loader = new NativeImageLoader(224, 224 ,3);
        INDArray image = loader.asMatrix(new File(path));
        //特征图可视化
        //DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        //scaler.transform(image);
        return image;
    }


    //INDArray转BufferedImage
    public static BufferedImage imageFromINDArray(INDArray array,int channel) {
        long[] shape = array.shape();
        long height = shape[2];
        long width = shape[3];
        BufferedImage image = new BufferedImage((int)width, (int)height, BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int red = array.getInt(0, channel, y, x);
                //int green = array.getInt(0, channel, y, x);
                //int blue = array.getInt(0, channel, y, x);
                //handle out of bounds pixel values
                red = Math.min(red, 255);
                //green = Math.min(green, 255);
                //blue = Math.min(blue, 255);
                red = Math.max(red, 0);
                //green = Math.max(green, 0);
                //blue = Math.max(blue, 0);
                //image.setRGB(x, y, new Color(red, green, blue).getRGB());
                image.setRGB(x, y, new Color(red).getRGB());
            }
        }
        return image;
    }

    public static void main(String[] args) throws Exception{

        //用IMAGENET初始权重保留VGG16原始结构
        ComputationGraph model = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.IMAGENET);
        System.out.println("model:"+model.summary());

        //测试图片
        INDArray in  = loadPic(new File("").getCanonicalPath()+"/JPEGImages/1.png");


        //取前面21层(一共就只有21层)
        Map<String, INDArray> result =  model.feedForward(in,20,false);

        //block1_conv1
        INDArray block1_conv1 = result.get("block1_conv1");
        System.out.println("block1_conv1:"+ Arrays.toString(block1_conv1.shape()));
        //block1_conv2
        INDArray block1_conv2 = result.get("block1_conv2");
        System.out.println("block1_conv2:"+ Arrays.toString(block1_conv2.shape()));
        //block1_pool
        INDArray block1_pool = result.get("block1_pool");
        System.out.println("block1_pool:"+ Arrays.toString(block1_pool.shape()));
        //block2_conv1
        INDArray block2_conv1 = result.get("block2_conv1");
        System.out.println("block2_conv1:"+ Arrays.toString(block2_conv1.shape()));
        //block2_conv2
        INDArray block2_conv2 = result.get("block2_conv2");
        System.out.println("block2_conv2:"+ Arrays.toString(block2_conv2.shape()));
        //block2_pool
        INDArray block2_pool = result.get("block2_pool");
        System.out.println("block2_pool:"+ Arrays.toString(block2_pool.shape()));
        //block3_conv1
        INDArray block3_conv1 = result.get("block3_conv1");
        System.out.println("block3_conv1:"+ Arrays.toString(block3_conv1.shape()));
        //block3_conv2
        INDArray block3_conv2 = result.get("block3_conv2");
        System.out.println("block3_conv2:"+ Arrays.toString(block3_conv2.shape()));
        //block3_conv3
        INDArray block3_conv3 = result.get("block3_conv3");
        System.out.println("block3_conv3:"+ Arrays.toString(block3_conv3.shape()));
        //block3_pool
        INDArray block3_pool = result.get("block3_pool");
        System.out.println("block3_pool:"+ Arrays.toString(block3_pool.shape()));
        //block4_conv1
        INDArray block4_conv1 = result.get("block4_conv1");
        System.out.println("block4_conv1:"+ Arrays.toString(block4_conv1.shape()));
        //block4_conv2
        INDArray block4_conv2 = result.get("block4_conv2");
        System.out.println("block4_conv2:"+ Arrays.toString(block4_conv2.shape()));
        //block4_conv3
        INDArray block4_conv3 = result.get("block4_conv3");
        System.out.println("block4_conv3:"+ Arrays.toString(block4_conv3.shape()));
        //block4_pool
        INDArray block4_pool = result.get("block4_pool");
        System.out.println("block4_pool:"+ Arrays.toString(block4_pool.shape()));
        //block5_conv1
        INDArray block5_conv1 = result.get("block5_conv1");
        System.out.println("block5_conv1:"+ Arrays.toString(block5_conv1.shape()));
        //block5_conv2
        INDArray block5_conv2 = result.get("block5_conv2");
        System.out.println("block5_conv2:"+ Arrays.toString(block5_conv2.shape()));
        //block5_conv3
        INDArray block5_conv3 = result.get("block5_conv3");
        System.out.println("block5_conv3:"+ Arrays.toString(block5_conv3.shape()));
        //block5_pool
        INDArray block5_pool = result.get("block5_pool");
        System.out.println("block5_pool:"+ Arrays.toString(block5_pool.shape()));
        //flatten
        INDArray flatten = result.get("flatten");
        System.out.println("flatten:"+ Arrays.toString(flatten.shape()));
        //fc1
        INDArray fc1 = result.get("fc1");
        System.out.println("fc1:"+ Arrays.toString(fc1.shape()));
        //fc2
        INDArray fc2 = result.get("fc2");
        System.out.println("fc2:"+ Arrays.toString(fc2.shape()));
        //predictions
        INDArray predictions = result.get("predictions");
        System.out.println("predictions:"+ Arrays.toString(predictions.shape()));

        //取某一层输出
        INDArray pic = block1_conv1;
        long channels = pic.shape()[1];
        System.out.println("channels:"+channels);
        //每个通道转为一张灰度图
        for(int i=0;i<=channels-1;i=i+5){
            BufferedImage image = imageFromINDArray(pic,i);
            //显示图片
            CanvasFrame canvas = new CanvasFrame("pic", i);
            canvas.showImage(image);
        }

    }




}
