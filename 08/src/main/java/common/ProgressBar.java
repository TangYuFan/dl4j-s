package common;

import javax.swing.*;
import javax.swing.border.Border;
import java.awt.*;

/**
 * 进度条
 * @author 小帅丶
 * by https://github.com/PacktPublishing/Java-Machine-Learning-for-Computer-Vision
 */
public class ProgressBar {

    private final JFrame mainFrame;
    private JProgressBar progressBar;
    private boolean unDecoreate = false;

    public ProgressBar(JFrame mainFrame) {
        this.mainFrame = mainFrame;
        progressBar = createProgressBar(mainFrame);
    }

    public ProgressBar(JFrame mainFrame, boolean unDecoreate) {
        this.mainFrame = mainFrame;
        progressBar = createProgressBar(mainFrame);
        this.unDecoreate = unDecoreate;
    }

    public void showProgressBar(String msg) {
        SwingUtilities.invokeLater(() -> {
            if (unDecoreate) {
                mainFrame.setUndecorated(true);
                mainFrame.setLocationRelativeTo(null);
            }
            progressBar = createProgressBar(mainFrame);
            progressBar.setString(msg);
            progressBar.setStringPainted(true);
            progressBar.setIndeterminate(true);
            progressBar.setVisible(true);
            mainFrame.add(progressBar, BorderLayout.NORTH);
            mainFrame.pack();
            mainFrame.setVisible(true);
            if (unDecoreate) {
                mainFrame.pack();
                mainFrame.setVisible(true);
            }
            mainFrame.repaint();
        });
    }


    private JProgressBar createProgressBar(JFrame mainFrame) {
    	Color color = new Color(90,200,250);
        JProgressBar jProgressBar = new JProgressBar(JProgressBar.HORIZONTAL);
        jProgressBar.setVisible(false);
        jProgressBar.setBackground(color);
        Border border = BorderFactory.createEmptyBorder() ;
		jProgressBar.setBorder(border);
        mainFrame.add(jProgressBar, BorderLayout.NORTH);
        return jProgressBar;
    }

    public void setVisible(boolean visible) {
        progressBar.setVisible(visible);
    }
}
