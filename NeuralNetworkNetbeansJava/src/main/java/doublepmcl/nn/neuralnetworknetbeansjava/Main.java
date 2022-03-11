package doublepmcl.nn.neuralnetworknetbeansjava;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * Created by haris on 1/29/16.
 */

// Exec from: http://alvinalexander.com/java/edu/pj/pj010016

public class Main {
    public static void main(String[] args) {
//        System.out.println("Hello World! Java!!!");
        System.out.println("Working Directory = " +
                System.getProperty("user.dir"));

        String[] cmd = {
                "python2.7",
                "src/PythonPrograms/Main.py"
        };

        String s;

        try {
            Process p = Runtime.getRuntime().exec(cmd);

            BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
            BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));

            while ((s = stdInput.readLine()) != null) {
                System.out.println(s);
            }

            while ((s = stdError.readLine()) != null) {
                System.out.println(s);
            }
        } catch (IOException e)  {
            System.out.println("IOExcaption happend:");
            e.printStackTrace();
            System.exit(-1);
        }
    }
}
