# LightPerceptron

The perceptron is an algorithm for supervised learning of binary classifiers. Perceptron was one of the first models of neural networks.

**LightPerceptron** is a simple Java library that allows you to create and train multilayer perceptrons.

Learning algorithm: backpropagation

Activation functions:
* SIGMOID: *f(x) = 1 / (1 - exp(-x))*

**LightPerceptron** also allows you to import / export trained neural networks into json files.

build from sources:
```
cd light-perceptron
mvn install
```

How to use:

Creating network with PerceptronBuilder:

```
        Perceptron perceptron = new PerceptronBuilder().
                addLayer(2, 2).
                addLayer(2, 1).
                build();
```
This code created perceptron with with 2 layers (hidden with 2 neurons and output layer with 1 neuron).

Sample 1: Animal classification task

| number of wings  | number of paws | has tail | insect | bird | snake | mammal | spider | 
|------------------|----------------|----------|--------|------|-------|--------|--------
|2 |6 |0 |1 |0 |0 |0 |0
|2 |2 |1 |0 |1 |0 |0 |0
|0 |0 |1 |0 |0 |1 |0 |0
|0 |4 |0 |0 |0 |0 |1 |0
|0 |8 |0 |0 |0 |0 |0 |1

    final double[][] DATA_ANIMALS = {
            {2, 6, 0, 1, 0, 0, 0, 0}, // insect
            {2, 2, 1, 0, 1, 0, 0, 0}, // bird
            {0, 0, 1, 0, 0, 1, 0, 0}, // snake
            {0, 4, 0, 0, 0, 0, 1, 0}, // mammal
            {0, 8, 0, 0, 0, 0, 0, 1}  // spider

    };
    PerceptronBuilder pb = new PerceptronBuilder();
    Perceptron p = pb.
            addLayer(3, 10).
            addLayer(10, 10).
            addLayer(10, 5).
            build();

    p.training(DATA_ANIMALS, 3, 5, 10000);
    double[] result = p.classification(new double[] {0, 4, 1});


Sample 2: Function approximation

```
        int maxIteration = 1000;
        PerceptronBuilder pb = new PerceptronBuilder();
        Perceptron p = pb.
                addLayer(2, 2).
                addLayer(2, 1).
                build();

        Random r = new Random();
        double a, b;

        for (int i = 0; i < maxIteration; i++) {
            // generate a, b, 0 <= a <= 1, 0 <= a <=
            a = r.nextDouble();
            b = r.nextDouble();
            p.training(
                    new double[]{a, b}, new double[]{Math.sin(a) * Math.cos(b)});
        }
        a = r.nextDouble();
        b = r.nextDouble();
        double[] calcResult = p.calculate(new double[] {a, b});
```

### Import / export to json format

sample code with import and export of perceptron state:
```
        final double[][] DATA_ANIMALS = {
            {2, 6, 0, 1, 0, 0, 0, 0}, // insect
            {2, 2, 1, 0, 1, 0, 0, 0}, // bird
            {0, 0, 1, 0, 0, 1, 0, 0}, // snake
            {0, 4, 0, 0, 0, 0, 1, 0}, // mammal
            {0, 8, 0, 0, 0, 0, 0, 1}  // spider
        };

        PerceptronBuilder pb = new PerceptronBuilder();
        Perceptron p = pb.
                addLayer(3, 10).
                addLayer(10, 10).
                addLayer(10, 5).
                build();


        // training neural net
        p.training(DATA_ANIMALS, 3, 5, 1000);
        
        double[] result = p.classification(new double[] {2, 6, 0});
        String str = p.exportToJson();

        Perceptron p2 = pb.loadFromJson(str);
        double[] result2 = p2.classification(new double[] {2, 6, 0});
```

Json with a perceptron state:

```
{
  "layers": [
    {
      "nInput": 3,
      "nOutput": 5,
      "neurons": [
        {
          "weights": [
            -1.7874804103060071,
            1.3755867550799334,
            -0.07769751578997851
          ],
          "threshold": -1.5545014269000352
        },
        {
          "weights": [
            -1.2657621818702498,
            -0.008026094042472627,
            1.817357363951599
          ],
          "threshold": 0.3079024314435577
        },
        {
          "weights": [
            2.3869309691959013,
            -1.6920546412863395,
            1.005298869863273
          ],
          "threshold": 1.5551406483323806
        },
        {
          "weights": [
            2.720545987044474,
            -0.05007972364363105,
            -2.1874063960889223
          ],
          "threshold": -1.1972588963891837
        },
        {
          "weights": [
            1.2029037228992203,
            1.76800111428798,
            -1.4732765710439788
          ],
          "threshold": -1.4070653175289247
        }
      ]
    },
    {
      "nInput": 5,
      "nOutput": 4,
      "neurons": [
        {
          "weights": [
            1.594427083203863,
            -1.8268179641073674,
            -3.6207943250994283,
            3.4972727771716685,
            -0.367092267344614
          ],
          "threshold": -2.1331673348798663
        },
        {
          "weights": [
            -2.9727129035197613,
            -2.1181851676309216,
            2.463672856513635,
            1.5202010079774575,
            1.7132122484985965
          ],
          "threshold": -2.71614972112006
        },
        {
          "weights": [
            -1.3486129625435426,
            0.6213533111488375,
            2.230717431733854,
            -2.0890523941773504,
            -3.491911158294164
          ],
          "threshold": 0.5714130803747863
        },
        {
          "weights": [
            1.7636905875821767,
            0.9143079497423531,
            -2.805789583056562,
            -4.097382327580715,
            0.8746795299254279
          ],
          "threshold": -1.001793049363282
        }
      ]
    }
  ]
}
```