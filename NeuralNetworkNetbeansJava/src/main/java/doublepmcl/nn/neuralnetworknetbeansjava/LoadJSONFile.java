package doublepmcl.nn.neuralnetworknetbeansjava;

import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.swing.text.StyledEditorKit;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Created by haris on 26.05.16.
 */
public class LoadJSONFile {

    static final Logger logger = LoggerFactory.getLogger(LoadJSONFile.class);

    private static final String workingPath = Paths.get(".").toAbsolutePath().normalize().toString();

    public static void println(String line) {
        //System.out.println(line);
        logger.info(line);
    }

    public static void main(String[] args) {
        println("working dir = "+workingPath);
//        File jsonFile = new File(workingPath+"/"+"dumpfile.json");
        String content = "";
        JSONParser parser = new JSONParser();
        try {
            content = new String(Files.readAllBytes(Paths.get(workingPath+"/jsonfiles/dumpfile.json")));

            Object obj = parser.parse(content);
            JSONArray jsonArray = (JSONArray)obj;
            double[][][] arrayData = new double[jsonArray.size()][][];

            for (int i = 0; i < jsonArray.size(); i++) {
                JSONArray array1 = (JSONArray)jsonArray.get(i);
                double[][] numberArray = new double[array1.size()][];
                for (int j = 0; j < array1.size(); j++) {
                    JSONArray array2 = (JSONArray)array1.get(j);
                    double[] numberArrayInner = new double[array2.size()];
                    for (int k = 0; k < array2.size(); k++) {
                        numberArrayInner[k] = Double.valueOf(array2.get(k).toString());
                    }
                    numberArray[j] = numberArrayInner;
                }
                arrayData[i] = numberArray;
            }
            println("parsing finished!!!");
            for (int i = 0; i < arrayData.length; i++) {
                println(Arrays.deepToString(arrayData[i]));
                BufferedImage bufferedImage = Utils.getImageFromArray(arrayData[i]);
                Utils.saveImg(bufferedImage, workingPath+"/pics/test_pic_from_array_"+i+".png");
                BufferedImage bufferedImageBigger = Utils.getScaledImage(bufferedImage, 10);
                Utils.saveImg(bufferedImageBigger, workingPath+"/pics/test_bigger_pic_from_array_"+i+".png");
            }
        } catch (Exception e) {

        }
    }
}
