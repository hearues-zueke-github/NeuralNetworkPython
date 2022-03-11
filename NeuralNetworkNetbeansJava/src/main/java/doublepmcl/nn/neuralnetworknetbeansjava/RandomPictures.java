/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package doublepmcl.nn.neuralnetworknetbeansjava;

import static doublepmcl.nn.neuralnetworknetbeansjava.Utils.getBufferedReader;
import static doublepmcl.nn.neuralnetworknetbeansjava.Utils.sysprintln;
import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;

/**
 *
 * @author haris
 */
public class RandomPictures extends javax.swing.JFrame implements KeyListener, WindowListener
{
    private final String[] namesOfSet = {"train", "valid", "test"};
    private final int[] amountsOfSet = {50, 10, 10};
    private String nameOfSet = "train";
    private int choosenSet = 0;
    
    private final int ZOOM_MIN = 1;
    private int ZOOM_MAX;
    private int zoomFactor = 10;
    
    private final boolean isMnistPicture;
    private BufferedImage originalMnistPicture;
    
    private final String absolutePath = Paths.get("").toAbsolutePath().toString();
    private final JFileChooser fileChooser = new JFileChooser(absolutePath);
    // <editor-fold defaultstate="collapsed" desc=" Variables declaration - do not modify ">
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnCreateRandomPicture;
    private javax.swing.JToggleButton btnDecrementIndex;
    private javax.swing.JToggleButton btnDecrementZoom;
    private javax.swing.JToggleButton btnIncrementIndex;
    private javax.swing.JToggleButton btnIncrementZoom;
    private javax.swing.JButton btnLoadChoosenData;
    private javax.swing.JButton btnLoadRandomMnistData;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JMenuItem jMenuItem1;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JLabel labelChooseWhichSet;
    private javax.swing.JLabel labelIndexOfSet;
    private javax.swing.JLabel labelPicture;
    private javax.swing.JLabel labelZoomPicture;
    private javax.swing.JMenu menuFile;
    private javax.swing.JMenu menuitemAbout;
    private javax.swing.JMenuItem menuitemExit;
    private javax.swing.JMenuItem menuitemFile;
    private javax.swing.JPanel panelMainButtons;
    private javax.swing.JRadioButton radbtnTestSet;
    private javax.swing.JRadioButton radbtnTrainSet;
    private javax.swing.JRadioButton radbtnValidSet;
    private javax.swing.ButtonGroup radbtngrpSet;
    private javax.swing.JTextField txtIndexOfSet;
    // End of variables declaration//GEN-END:variables
    //</editor-fold>
    /**
     * Creates new form ShowNumbersGUI
     */
    public RandomPictures()
    {
        this.isMnistPicture = false;
        this.ZOOM_MAX = 15;
//        GlobalScreen.setEventDispatcher(new SwingDispatchService());
        addWindowListener(this);
        addKeyListener(this);
        setFocusable(true);
        
        addFocusListener(new FocusAdapter() {

        /**
         * {@inheritDoc}
         */
        @Override
        public void focusGained(FocusEvent aE) {
            RandomPictures.this.requestFocusInWindow();
        }
    });
        fileChooser.setPreferredSize(new Dimension(fileChooser.getPreferredSize().width + 100, fileChooser.getPreferredSize().height + 100));
        initComponents();
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String args[])
    {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html
         */
        try
        {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels())
            {
                if ("Nimbus".equals(info.getName()))
                {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        }
        catch (ClassNotFoundException ex)
        {
            java.util.logging.Logger.getLogger(RandomPictures.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        catch (InstantiationException ex)
        {
            java.util.logging.Logger.getLogger(RandomPictures.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        catch (IllegalAccessException ex)
        {
            java.util.logging.Logger.getLogger(RandomPictures.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        catch (javax.swing.UnsupportedLookAndFeelException ex)
        {
            java.util.logging.Logger.getLogger(RandomPictures.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable()
        {
            public void run()
            {
                new RandomPictures().setVisible(true);
            }
        });
//        SwingUtilities.invokeLater(new Runnable() {
//            public void run() {
//                new RandomPictures().setVisible(true);
//            }
//        });
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        radbtngrpSet = new javax.swing.ButtonGroup();
        panelMainButtons = new javax.swing.JPanel();
        btnLoadChoosenData = new javax.swing.JButton();
        txtIndexOfSet = new javax.swing.JTextField();
        labelIndexOfSet = new javax.swing.JLabel();
        radbtnTrainSet = new javax.swing.JRadioButton();
        radbtnValidSet = new javax.swing.JRadioButton();
        radbtnTestSet = new javax.swing.JRadioButton();
        labelChooseWhichSet = new javax.swing.JLabel();
        btnLoadRandomMnistData = new javax.swing.JButton();
        btnIncrementIndex = new javax.swing.JToggleButton();
        btnDecrementZoom = new javax.swing.JToggleButton();
        btnCreateRandomPicture = new javax.swing.JButton();
        btnDecrementIndex = new javax.swing.JToggleButton();
        btnIncrementZoom = new javax.swing.JToggleButton();
        labelZoomPicture = new javax.swing.JLabel();
        jPanel1 = new javax.swing.JPanel();
        labelPicture = new javax.swing.JLabel();
        jMenuBar1 = new javax.swing.JMenuBar();
        menuFile = new javax.swing.JMenu();
        menuitemFile = new javax.swing.JMenuItem();
        menuitemExit = new javax.swing.JMenuItem();
        menuitemAbout = new javax.swing.JMenu();
        jMenuItem1 = new javax.swing.JMenuItem();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Random Picture Generator");
        setMinimumSize(new java.awt.Dimension(710, 490));
        setPreferredSize(new java.awt.Dimension(710, 490));
        getContentPane().setLayout(new org.netbeans.lib.awtextra.AbsoluteLayout());

        panelMainButtons.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED));
        panelMainButtons.setLayout(new org.netbeans.lib.awtextra.AbsoluteLayout());

        btnLoadChoosenData.setText("Load Choosen Data");
        btnLoadChoosenData.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnLoadChoosenDataActionPerformed(evt);
            }
        });
        panelMainButtons.add(btnLoadChoosenData, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 240, 230, 30));

        txtIndexOfSet.setHorizontalAlignment(javax.swing.JTextField.CENTER);
        txtIndexOfSet.setText("0");
        txtIndexOfSet.setCursor(new java.awt.Cursor(java.awt.Cursor.TEXT_CURSOR));
        txtIndexOfSet.setPreferredSize(new java.awt.Dimension(100, 23));
        txtIndexOfSet.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                txtIndexOfSetKeyPressed(evt);
            }
        });
        panelMainButtons.add(txtIndexOfSet, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 210, 130, -1));

        labelIndexOfSet.setText("Index of Set");
        panelMainButtons.add(labelIndexOfSet, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 190, -1, -1));

        radbtngrpSet.add(radbtnTrainSet);
        radbtnTrainSet.setSelected(true);
        radbtnTrainSet.setText("Train Set");
        radbtnTrainSet.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                radbtnTrainSetActionPerformed(evt);
            }
        });
        panelMainButtons.add(radbtnTrainSet, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 100, -1, -1));

        radbtngrpSet.add(radbtnValidSet);
        radbtnValidSet.setText("Valid Set");
        radbtnValidSet.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                radbtnValidSetActionPerformed(evt);
            }
        });
        panelMainButtons.add(radbtnValidSet, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 130, -1, -1));

        radbtngrpSet.add(radbtnTestSet);
        radbtnTestSet.setText("Test Set");
        radbtnTestSet.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                radbtnTestSetActionPerformed(evt);
            }
        });
        panelMainButtons.add(radbtnTestSet, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 160, -1, -1));

        labelChooseWhichSet.setText("Choose which Set to show");
        panelMainButtons.add(labelChooseWhichSet, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 80, -1, -1));

        btnLoadRandomMnistData.setText("Load Random Mnist Data");
        btnLoadRandomMnistData.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnLoadRandomMnistDataActionPerformed(evt);
            }
        });
        panelMainButtons.add(btnLoadRandomMnistData, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 40, 230, 30));

        btnIncrementIndex.setFont(new java.awt.Font("Dialog", 1, 18)); // NOI18N
        btnIncrementIndex.setText("+");
        btnIncrementIndex.setBorder(null);
        btnIncrementIndex.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnIncrementIndexActionPerformed(evt);
            }
        });
        panelMainButtons.add(btnIncrementIndex, new org.netbeans.lib.awtextra.AbsoluteConstraints(200, 190, 40, 40));

        btnDecrementZoom.setFont(new java.awt.Font("Dialog", 1, 18)); // NOI18N
        btnDecrementZoom.setText("-");
        btnDecrementZoom.setBorder(null);
        btnDecrementZoom.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnDecrementZoomActionPerformed(evt);
            }
        });
        panelMainButtons.add(btnDecrementZoom, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 300, 40, 40));

        btnCreateRandomPicture.setText("Create Random Picture");
        btnCreateRandomPicture.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnCreateRandomPictureActionPerformed(evt);
            }
        });
        panelMainButtons.add(btnCreateRandomPicture, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 10, 230, -1));
        btnCreateRandomPicture.getAccessibleContext().setAccessibleParent(panelMainButtons);

        btnDecrementIndex.setFont(new java.awt.Font("Dialog", 1, 18)); // NOI18N
        btnDecrementIndex.setText("-");
        btnDecrementIndex.setBorder(null);
        btnDecrementIndex.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnDecrementIndexActionPerformed(evt);
            }
        });
        panelMainButtons.add(btnDecrementIndex, new org.netbeans.lib.awtextra.AbsoluteConstraints(150, 190, 40, 40));

        btnIncrementZoom.setFont(new java.awt.Font("Dialog", 1, 18)); // NOI18N
        btnIncrementZoom.setText("+");
        btnIncrementZoom.setBorder(null);
        btnIncrementZoom.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnIncrementZoomActionPerformed(evt);
            }
        });
        panelMainButtons.add(btnIncrementZoom, new org.netbeans.lib.awtextra.AbsoluteConstraints(60, 300, 40, 40));

        labelZoomPicture.setText("Zoom Picture");
        panelMainButtons.add(labelZoomPicture, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 280, -1, -1));

        getContentPane().add(panelMainButtons, new org.netbeans.lib.awtextra.AbsoluteConstraints(440, 10, 250, 420));

        jPanel1.setLayout(new java.awt.GridLayout(1, 1));

        labelPicture.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        labelPicture.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        jPanel1.add(labelPicture);

        getContentPane().add(jPanel1, new org.netbeans.lib.awtextra.AbsoluteConstraints(10, 10, 420, 420));

        menuFile.setText("File");

        menuitemFile.setAccelerator(javax.swing.KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_N, java.awt.event.InputEvent.CTRL_MASK));
        menuitemFile.setText("File");
        menuFile.add(menuitemFile);

        menuitemExit.setAccelerator(javax.swing.KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_W, java.awt.event.InputEvent.CTRL_MASK));
        menuitemExit.setText("Exit");
        menuFile.add(menuitemExit);

        jMenuBar1.add(menuFile);

        menuitemAbout.setText("Help");

        jMenuItem1.setAccelerator(javax.swing.KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_H, java.awt.event.InputEvent.CTRL_MASK));
        jMenuItem1.setText("About");
        jMenuItem1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jMenuItem1ActionPerformed(evt);
            }
        });
        menuitemAbout.add(jMenuItem1);

        jMenuBar1.add(menuitemAbout);

        setJMenuBar(jMenuBar1);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jMenuItem1ActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_jMenuItem1ActionPerformed
    {//GEN-HEADEREND:event_jMenuItem1ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_jMenuItem1ActionPerformed

    private void btnCreateRandomPictureActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnCreateRandomPictureActionPerformed
        // TODO add your handling code here:
        try {
            Random rand = new Random();
            int width = 400;
            int height = 400;
            BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            for (int j =0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    img.setRGB(i, j, new Color(Math.abs(rand.nextInt()) % 256, Math.abs(rand.nextInt()) % 256, Math.abs(rand.nextInt()) % 256).getRGB());
                }
            }
            ImageIcon imageIcon = new ImageIcon(img);
            labelPicture.setIcon(imageIcon);
            File outputfile = new File(absolutePath+"/images/test_random.png");
//            ImageIO.write(img, "png", outputfile);
        } catch (Exception e) {

        }
    }//GEN-LAST:event_btnCreateRandomPictureActionPerformed

    private void btnLoadChoosenDataActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnLoadChoosenDataActionPerformed
        // TODO add your handling code here:
        int index = Integer.parseInt(txtIndexOfSet.getText());

        if (index < 0) {
            txtIndexOfSet.setText(Integer.toString(0));
            index = 0;
        } else if (index >= amountsOfSet[choosenSet]*1000) {
            txtIndexOfSet.setText(Integer.toString(amountsOfSet[choosenSet]*1000 - 1));
            index = amountsOfSet[choosenSet]*1000 - 1;
        }
        
        int indexFile = index / 1000;
        int indexOffset = index % 1000;
        String filePath = absolutePath + "/jsonfiles/mnist_path/"+nameOfSet+"_path/"+nameOfSet+"_"+indexFile+".json.gz";
        try {
            BufferedReader jsonFileReader = getBufferedReader(filePath);
            String content = "";
            String line;
            while ((line = jsonFileReader.readLine()) != null) {
                content += line;
            }
            JSONParser parser = new JSONParser();

            Object obj = parser.parse(content);
            JSONArray jsonArray = (JSONArray)((JSONArray)obj).get(0);
            double[][][] arrayData = new double[jsonArray.size()][][];
            
            sysprintln("before iterating");
            for (int i = 0; i < jsonArray.size(); i++) {
                JSONArray array1 = (JSONArray)jsonArray.get(i);
                double[][] numberArray = new double[28][];
                for (int j = 0; j < 28; j++) {
                    double[] numberArrayInner = new double[28];
                    for (int k = 0; k < 28; k++) {
                        numberArrayInner[k] = Double.valueOf(array1.get(j+k*28).toString());
                    }
                    numberArray[j] = numberArrayInner;
                }
                arrayData[i] = numberArray;
            }
            sysprintln("parsing finished!!!");
            double[][] choosenArray = arrayData[indexOffset];
            originalMnistPicture = Utils.getImageFromArray(choosenArray);
            BufferedImage bufferedImageBigger = Utils.getScaledImage(originalMnistPicture, zoomFactor);
            ImageIcon icon = new ImageIcon(bufferedImageBigger);
            labelPicture.setIcon(icon);
        } catch (Exception e) {
            System.out.println(Arrays.deepToString(e.getStackTrace()));
        }
    }//GEN-LAST:event_btnLoadChoosenDataActionPerformed

    private void radbtnTrainSetActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_radbtnTrainSetActionPerformed
    {//GEN-HEADEREND:event_radbtnTrainSetActionPerformed
        nameOfSet = "train";
        choosenSet = 0;
    }//GEN-LAST:event_radbtnTrainSetActionPerformed

    private void radbtnValidSetActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_radbtnValidSetActionPerformed
    {//GEN-HEADEREND:event_radbtnValidSetActionPerformed
        nameOfSet = "valid";
        choosenSet = 1;
    }//GEN-LAST:event_radbtnValidSetActionPerformed

    private void radbtnTestSetActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_radbtnTestSetActionPerformed
    {//GEN-HEADEREND:event_radbtnTestSetActionPerformed
        nameOfSet = "test";
        choosenSet = 2;
    }//GEN-LAST:event_radbtnTestSetActionPerformed

    private void btnLoadRandomMnistDataActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_btnLoadRandomMnistDataActionPerformed
    {//GEN-HEADEREND:event_btnLoadRandomMnistDataActionPerformed
        // TODO add your handling code here:
//        FileInputStream fin = new FileInputStream(FILENAME);
//        GZIPInputStream gzis = new GZIPInputStream(fin);
//        InputStreamReader xover = new InputStreamReader(gzis);
//        BufferedReader is = new BufferedReader(xover);
        Random random = new Random();
        int randomSet = Math.abs(random.nextInt()) % namesOfSet.length;
        System.out.println("randomSet = "+randomSet);
        
        String filePath = absolutePath+"/jsonfiles/mnist_path/"+namesOfSet[randomSet]+"_path/"+namesOfSet[randomSet]+"_"+Math.abs(random.nextInt())%amountsOfSet[randomSet]+".json.gz";
        System.out.println("filePath = "+filePath);
        sysprintln("Is Working!!!");
        try {
            BufferedReader jsonFileReader = getBufferedReader(filePath);
            String content = "";
            String line;
            while ((line = jsonFileReader.readLine()) != null) {
                content += line;
            }
//            sysprintln(content);
            JSONParser parser = new JSONParser();
//            content = new String(Files.readAllBytes(Paths.get(workingPath+"/jsonfiles/dumpfile.json")));

//            sysprintln(String.valueOf(Thread.currentThread().getStackTrace()[0].getLineNumber()));
            Object obj = parser.parse(content);
            JSONArray jsonArray = (JSONArray)((JSONArray)obj).get(0);
            double[][][] arrayData = new double[jsonArray.size()][][];
            
            sysprintln("before iterating");
            for (int i = 0; i < jsonArray.size(); i++) {
//                sysprintln("i = "+i);
                JSONArray array1 = (JSONArray)jsonArray.get(i);
                double[][] numberArray = new double[28][];
                for (int j = 0; j < 28; j++) {
//                    sysprintln("j = "+j);
//                    JSONArray array2 = (JSONArray)array1.get(j);
                    double[] numberArrayInner = new double[28];
                    for (int k = 0; k < 28; k++) {
//                        sysprintln("k = "+k);
                        numberArrayInner[k] = Double.valueOf(array1.get(j+k*28).toString());
                    }
                    numberArray[j] = numberArrayInner;
                }
                arrayData[i] = numberArray;
            }
            sysprintln("parsing finished!!!");
            double[][] choosenArray = arrayData[Math.abs(random.nextInt()) % 1000];
//            sysprintln(Arrays.deepToString(choosenArray));
            originalMnistPicture = Utils.getImageFromArray(choosenArray);
            BufferedImage bufferedImageBigger = Utils.getScaledImage(originalMnistPicture, zoomFactor);
            ImageIcon icon = new ImageIcon(bufferedImageBigger);
            labelPicture.setIcon(icon);

        } catch (Exception e) {
            System.out.println();
        }
    }//GEN-LAST:event_btnLoadRandomMnistDataActionPerformed

    private void btnDecrementZoomActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnDecrementZoomActionPerformed
        // TODO add your handling code here:
        if (--zoomFactor < ZOOM_MIN)
            zoomFactor = ZOOM_MIN;
        
        BufferedImage bufferedImageBigger = Utils.getScaledImage(originalMnistPicture, zoomFactor);
        ImageIcon icon = new ImageIcon(bufferedImageBigger);
        labelPicture.setIcon(icon);
    }//GEN-LAST:event_btnDecrementZoomActionPerformed

    private void btnIncrementIndexActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnIncrementIndexActionPerformed
        // TODO add your handling code here:
        txtIndexOfSet.setText(String.valueOf(Integer.parseInt(txtIndexOfSet.getText())+1));
        ActionListener[] actions = btnLoadChoosenData.getActionListeners();
        for (ActionListener action : actions) {
            action.actionPerformed(new ActionEvent(this, ActionEvent.ACTION_PERFORMED, null){});
        }
    }//GEN-LAST:event_btnIncrementIndexActionPerformed

    private void txtIndexOfSetKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_txtIndexOfSetKeyPressed
        // TODO add your handling code here:
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            ActionListener[] actions = btnLoadChoosenData.getActionListeners();
            for (ActionListener action : actions) {
                action.actionPerformed(new ActionEvent(this, ActionEvent.ACTION_PERFORMED, null){});
            }   
        }
    }//GEN-LAST:event_txtIndexOfSetKeyPressed

    private void btnDecrementIndexActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnDecrementIndexActionPerformed
        // TODO add your handling code here:
        txtIndexOfSet.setText(String.valueOf(Integer.parseInt(txtIndexOfSet.getText())-1));
        ActionListener[] actions = btnLoadChoosenData.getActionListeners();
        for (ActionListener action : actions) {
            action.actionPerformed(new ActionEvent(this, ActionEvent.ACTION_PERFORMED, null){});
        } 
    }//GEN-LAST:event_btnDecrementIndexActionPerformed

    private void btnIncrementZoomActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnIncrementZoomActionPerformed
        // TODO add your handling code here:
        if (++zoomFactor > ZOOM_MAX)
            zoomFactor = ZOOM_MAX;
        
        BufferedImage bufferedImageBigger = Utils.getScaledImage(originalMnistPicture, zoomFactor);
        ImageIcon icon = new ImageIcon(bufferedImageBigger);
        labelPicture.setIcon(icon);
    }//GEN-LAST:event_btnIncrementZoomActionPerformed
    // End of variables declaration
    
    public void windowOpened(WindowEvent e) { /* Unimplemented */ }
    public void windowClosed(WindowEvent e) { /* Unimplemented */ }
    public void windowClosing(WindowEvent e) { /* Unimplemented */ }
    public void windowIconified(WindowEvent e) { /* Unimplemented */ }
    public void windowDeiconified(WindowEvent e) { /* Unimplemented */ }
    public void windowActivated(WindowEvent e) { /* Unimplemented */ }
    public void windowDeactivated(WindowEvent e) { /* Unimplemented */ }

    @Override
    public void keyTyped(KeyEvent e) {
        System.out.println("keyTyped: key = "+e.getKeyChar());
    }

    @Override
    public void keyPressed(KeyEvent e) {
        System.out.println("keyPressed: key = "+e.getKeyChar());
    }

    @Override
    public void keyReleased(KeyEvent e) {
        System.out.println("keyReleased: key = "+e.getKeyChar());
    }
}
