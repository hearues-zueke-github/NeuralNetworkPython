package doublepmcl.nn.neuralnetworknetbeansjava;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by haris on 7/26/16.
 */
public class TestStreams {

    public static class Foo {
        public int age;
        public String name;

        Foo(int age, String name ) {
            this.age = age;
            this.name = name;
        }
    }


    public static void main(String[] args) {
        System.out.println("Hello World!");

        List<Foo> fooList = new ArrayList<>();
        fooList.add(new Foo(1, "one"));
        fooList.add(new Foo(2, "two"));
        fooList.add(new Foo(3, "three"));
        fooList.add(new Foo(4,  "four"));
        fooList.add(new Foo(5, "five"));
        fooList.add(new Foo(6, "six"));
        fooList.add(new Foo(7, "seven"));
        fooList.add(new Foo(8, "eight"));
        fooList.add(new Foo(9, "nine"));

        System.out.println(Arrays.toString(
                fooList.stream().parallel().filter(f -> f.age > 5).parallel().map(f -> f.name).parallel().collect(Collectors.toList()).toArray()
        )
        );//.forEachOrdered(System.out::println);

        System.out.println(Arrays.asList("A,B,C,D,E,F".split(",")).stream().reduce("", (a, b) -> a + b + b));//.forEachOrdered(System.out::println);
        System.out.println(IntStream.range(0, 101).parallel().reduce(0, (a, b) -> a + b));
//        IntStream.range(0,10000001).parallel().forEachOrdered( e -> System.out.println( "--> "+e));
//        IntStream.range(0,10000001).forEachOrdered( e -> System.out.println( "--> "+e));
    }
}
