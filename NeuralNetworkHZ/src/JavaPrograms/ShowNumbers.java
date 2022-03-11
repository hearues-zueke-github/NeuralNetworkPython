package JavaPrograms;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Created by haris on 5/24/16.
 */
public class ShowNumbers extends JFrame {
    JPanel panel = null;
    JLabel label = null;
    JButton button = null;

    DrawNumbers drawNumbers = null;
    public ShowNumbers(String title) {
//        super();
        this.setTitle(title);
        this.setLayout(new BorderLayout());//FlowLayout());

        panel = new JPanel();
        panel.setSize(new Dimension(200, 200));
        label = new JLabel();
        label.setText("This is a label!");
        button = new JButton();
        button.setText("Press me");

//        panel.add(label);
//        panel.add(button);
//        this.add(panel);
        drawNumbers = new DrawNumbers();
        drawNumbers.btnCopyText.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                // display/center the jdialog when the button is pressed
//                JDialog d = new JDialog(frame, "Hello", true);
//                d.setLocationRelativeTo(frame);
//                d.setVisible(true);
                drawNumbers.txtField2.setText(drawNumbers.txtField1.getText());
                drawNumbers.$$$getRootComponent$$$().setBackground(Color.cyan);
                ShowNumbers.this.setBackground(Color.green);
            }
        });
        drawNumbers.$$$getRootComponent$$$().setBorder(new EmptyBorder(10, 10, 10, 10));
        this.setContentPane(drawNumbers.$$$getRootComponent$$$());

//        Component rigidAreaWest = Box.createRigidArea(new Dimension(10, 0));
//        Component rigidAreaEast = Box.createRigidArea(new Dimension(20, 0));
//        Component rigidAreaNorth = Box.createRigidArea(new Dimension(0, 30));
//        Component rigidAreaSouth = Box.createRigidArea(new Dimension(0, 40));
//
//        rigidAreaWest.setBackground(new Color(0, 0, 0, 0));
//        rigidAreaEast.setBackground(new Color(0, 0, 0, 0));
//        rigidAreaNorth.setBackground(new Color(0, 0, 0, 0));
//        rigidAreaSouth.setBackground(new Color(0, 0, 0, 0));
//
//        this.add(rigidAreaWest, BorderLayout.WEST);
//        this.add(rigidAreaEast, BorderLayout.EAST);
//        this.add(rigidAreaNorth, BorderLayout.NORTH);
//        this.add(rigidAreaSouth, BorderLayout.SOUTH);

        this.setSize(300, 400);
        this.setLocationRelativeTo(null);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setVisible(false);
    }

    public static void main(String[] args) {
        JFrame frame = new ShowNumbers("JFrame Example");

//        frame.setSize(new Dimension(400, 300));
        frame.setVisible(true);
    }
}
