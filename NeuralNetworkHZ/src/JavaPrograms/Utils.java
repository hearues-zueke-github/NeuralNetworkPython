package JavaPrograms;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.File;
import java.nio.Buffer;

/**
 * Created by haris on 26.05.16.
 */
public class Utils {
    public static BufferedImage getImageFromArray(double[][] array) {
        BufferedImage bufferedImage = new BufferedImage(array[0].length, array.length, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < array.length; y++) {
            for (int x = 0; x < array[y].length; x++) {
                int value = (int)(array[y][x] * 255);
                bufferedImage.setRGB(x, y, new Color(value, value, value).getRGB());
            }
        }

        return bufferedImage;
    }

    public static BufferedImage getScaledImage(BufferedImage buffImg, int scale) {
        int width = buffImg.getWidth();
        int height = buffImg.getHeight();
        BufferedImage imgBigger = new BufferedImage(buffImg.getWidth()*scale, buffImg.getHeight()*scale, BufferedImage.TYPE_INT_RGB);

        for (int j =0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                for (int jk = 0; jk < scale; jk++) {
                    for (int ji = 0; ji < scale; ji++) {
                        imgBigger.setRGB(i*scale+ji, j*scale+jk, buffImg.getRGB(i, j));
                    }
                }
            }
        }

        return imgBigger;
    }

    public static void saveImg(BufferedImage bufferedImage, String path) {
        try {
            ImageIO.write(bufferedImage, "PNG", new File(path));
        } catch (Exception e) {
            System.out.println("Error when trying to write Image to path: "+path+" !!!");
        }
    }
}
