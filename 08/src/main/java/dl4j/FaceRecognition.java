package dl4j;


import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Slf4j
public class FaceRecognition {
	//阈值
    private static final double THRESHOLD = 0.25;
    private FaceNetSmallV2Model faceNetSmallV2Model;
    private ComputationGraph computationGraph;
    private static final NativeImageLoader LOADER = new NativeImageLoader(96, 96, 3);
    private final HashMap<String, INDArray> memberEncodingsMap = new HashMap<>();

    private INDArray transpose(INDArray indArray1) {
        INDArray one = Nd4j.create(new int[]{1, 96, 96});
        one.assign(indArray1.get(NDArrayIndex.point(0), NDArrayIndex.point(2)));
        INDArray two = Nd4j.create(new int[]{1, 96, 96});
        two.assign(indArray1.get(NDArrayIndex.point(0), NDArrayIndex.point(1)));
        INDArray three = Nd4j.create(new int[]{1, 96, 96});
        three.assign(indArray1.get(NDArrayIndex.point(0), NDArrayIndex.point(0)));
        return Nd4j.concat(0, one, two, three).reshape(new int[]{1, 3, 96, 96});
    }

    private INDArray read(String pathname) throws IOException {
        opencv_core.Mat imread = opencv_imgcodecs.imread(new File(pathname).getAbsolutePath(), 1);
        INDArray indArray = LOADER.asMatrix(imread);
        return transpose(indArray);
    }

    //编码
    private INDArray forwardPass(INDArray indArray) {
        Map<String, INDArray> output = computationGraph.feedForward(indArray, false);
        GraphVertex embeddings = computationGraph.getVertex("encodings");
        INDArray dense = output.get("dense");
        embeddings.setInputs(dense);
        INDArray embeddingValues = embeddings.doForward(false, LayerWorkspaceMgr.builder().defaultNoWorkspace().build());
        System.out.println("密集度 =                 " + dense);
        System.out.println("编码值 =                 " + embeddingValues);
        return embeddingValues;
    }

    //计算距离
    private double distance(INDArray a, INDArray b) {
        //2范数(欧氏距离)
        return a.distance2(b);
    }

    public void loadModel() throws Exception {
        faceNetSmallV2Model = new FaceNetSmallV2Model();
        computationGraph = faceNetSmallV2Model.init();
        System.out.println(computationGraph.summary());
    }


    //添加新脸
    public void registerNewMember(String memberId, String imagePath) throws IOException {
        INDArray read = read(imagePath);
        memberEncodingsMap.put(memberId, forwardPass(normalize(read)));
    }

    private static INDArray normalize(INDArray read) {
        return read.div(255.0);
    }


    //人脸匹配
    public String whoIs(String imagePath) throws IOException {
        INDArray read = read(imagePath);
        INDArray encodings = forwardPass(normalize(read));
        double minDistance = Double.MAX_VALUE;
        String foundUser = "";
        //和人脸库中已知编码计算2范数
        for (Map.Entry<String, INDArray> entry : memberEncodingsMap.entrySet()) {
            INDArray value = entry.getValue();
            double distance = distance(value, encodings);
            System.out.println("distance of " + entry.getKey() + " with " + new File(imagePath).getName() + " is " + distance);
            if (distance < minDistance) {
                minDistance = distance;
                foundUser = entry.getKey();
            }
        }
        //阈值取0.25
//        if (minDistance > THRESHOLD) {
//            foundUser = "貌似没有Ta";
//        }
        System.out.println(foundUser + " with distance " + minDistance);
        return foundUser;
    }

    public static void main(String[] args) throws Exception{

        FaceRecognition recognition = new FaceRecognition();
        recognition.loadModel();

        //注册人脸
        //Al_Sharpton
        recognition.registerNewMember("Al_Sharpton_1","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Al_Sharpton\\Al_Sharpton_0001.jpg");
        recognition.registerNewMember("Al_Sharpton_2","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Al_Sharpton\\Al_Sharpton_0002.jpg");
        recognition.registerNewMember("Al_Sharpton_3","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Al_Sharpton\\Al_Sharpton_0003.jpg");
        recognition.registerNewMember("Al_Sharpton_4","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Al_Sharpton\\Al_Sharpton_0004.jpg");
        recognition.registerNewMember("Al_Sharpton_5","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Al_Sharpton\\Al_Sharpton_0005.jpg");
        recognition.registerNewMember("Al_Sharpton_6","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Al_Sharpton\\Al_Sharpton_0006.jpg");
        //Ariel_Sharon
        recognition.registerNewMember("Ariel_Sharon_1","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Ariel_Sharon\\Ariel_Sharon_0001.jpg");
        recognition.registerNewMember("Ariel_Sharon_2","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Ariel_Sharon\\Ariel_Sharon_0002.jpg");
        recognition.registerNewMember("Ariel_Sharon_3","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Ariel_Sharon\\Ariel_Sharon_0003.jpg");
        recognition.registerNewMember("Ariel_Sharon_4","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Ariel_Sharon\\Ariel_Sharon_0004.jpg");
        recognition.registerNewMember("Ariel_Sharon_5","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Ariel_Sharon\\Ariel_Sharon_0005.jpg");
        recognition.registerNewMember("Ariel_Sharon_6","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Ariel_Sharon\\Ariel_Sharon_0006.jpg");
        //Arnold_Schwarzenegger
        recognition.registerNewMember("Arnold_Schwarzenegger_1","C:\\Users\\2019\\Desktop\\images\\Arnold_Schwarzenegger\\Arnold_Schwarzenegger_0001.jpg");
        recognition.registerNewMember("Arnold_Schwarzenegger_2","C:\\Users\\2019\\Desktop\\images\\Arnold_Schwarzenegger\\Arnold_Schwarzenegger_0002.jpg");
        recognition.registerNewMember("Arnold_Schwarzenegger_3","C:\\Users\\2019\\Desktop\\images\\Arnold_Schwarzenegger\\Arnold_Schwarzenegger_0003.jpg");
        recognition.registerNewMember("Arnold_Schwarzenegger_4","C:\\Users\\2019\\Desktop\\images\\Arnold_Schwarzenegger\\Arnold_Schwarzenegger_0004.jpg");
        recognition.registerNewMember("Arnold_Schwarzenegger_5","C:\\Users\\2019\\Desktop\\images\\Arnold_Schwarzenegger\\Arnold_Schwarzenegger_0005.jpg");
        recognition.registerNewMember("Arnold_Schwarzenegger_6","C:\\Users\\2019\\Desktop\\images\\Arnold_Schwarzenegger\\Arnold_Schwarzenegger_0006.jpg");
        recognition.registerNewMember("Arnold_Schwarzenegger_7","C:\\Users\\2019\\Desktop\\images\\Arnold_Schwarzenegger\\Arnold_Schwarzenegger_0007.jpg");
        //George_W_Bush
        recognition.registerNewMember("George_W_Bush_1","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\George_W_Bush\\George_W_Bush_0001.jpg");
        recognition.registerNewMember("George_W_Bush_2","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\George_W_Bush\\George_W_Bush_0002.jpg");
        recognition.registerNewMember("George_W_Bush_3","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\George_W_Bush\\George_W_Bush_0003.jpg");
        recognition.registerNewMember("George_W_Bush_4","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\George_W_Bush\\George_W_Bush_0004.jpg");
        //Gerhard_Schroeder
        recognition.registerNewMember("Gerhard_Schroeder_1","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Gerhard_Schroeder\\Gerhard_Schroeder_0001.jpg");
        recognition.registerNewMember("Gerhard_Schroeder_2","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Gerhard_Schroeder\\Gerhard_Schroeder_0002.jpg");
        recognition.registerNewMember("Gerhard_Schroeder_3","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Gerhard_Schroeder\\Gerhard_Schroeder_0003.jpg");
        recognition.registerNewMember("Gerhard_Schroeder_4","C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Gerhard_Schroeder\\Gerhard_Schroeder_0004.jpg");


        //识别
        String res1 = recognition.whoIs("C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Al_Sharpton\\Al_Sharpton_0007.jpg");
        String res2 = recognition.whoIs("C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Ariel_Sharon\\Ariel_Sharon_0007.jpg");
        String res3 = recognition.whoIs("C:\\Users\\2019\\Desktop\\images\\Arnold_Schwarzenegger\\Arnold_Schwarzenegger_0008.jpg");
        String res4 = recognition.whoIs("C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\George_W_Bush\\George_W_Bush_0005.jpg");
        String res5 = recognition.whoIs("C:\\Users\\2019\\Desktop\\tem\\src\\main\\resources\\people\\Gerhard_Schroeder\\Gerhard_Schroeder_0005.jpg");

        System.out.println("Al_Sharpton is recognition to "+res1);
        System.out.println("Ariel_Sharon is recognition to "+res2);
        System.out.println("Arnold_Schwarzenegger is recognition to "+res3);
        System.out.println("George_W_Bush is recognition to "+res4);
        System.out.println("Gerhard_Schroeder is recognition to "+res5);

    }


}
