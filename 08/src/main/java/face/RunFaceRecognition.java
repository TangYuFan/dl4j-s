package face;

import cn.xsshome.recogntion.face.ui.FaceRecogntionUI;
import common.ProgressBar;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.util.concurrent.Executors;


/**
 * Created by Klevis Ramo
 * 运行Demo
 */
public class RunFaceRecognition {
	private final static Logger LOGGER = LoggerFactory.getLogger(RunFaceRecognition.class);
    public static void main(String[] args) throws Exception {
        JFrame mainFrame = new JFrame();
        ProgressBar progressBar = new ProgressBar(mainFrame, true);
        progressBar.showProgressBar("Loading model, this make take few moments...");
        FaceRecogntionUI faceRecogntionUi = new FaceRecogntionUI();
        Executors.newCachedThreadPool().submit(() -> {
            try {
                faceRecogntionUi.initUI();
            } catch (Exception e) {
                e.printStackTrace();
                LOGGER.error("Failed to start ",e);
                throw new RuntimeException(e);
            } finally {
                progressBar.setVisible(false);
                mainFrame.dispose();
            }
        });

    }

}
