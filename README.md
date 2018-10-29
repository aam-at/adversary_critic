# Improved Network Robustness with Adversary Critic

[Alexander Matyasko](https://github.com/aam-at), Lap-Pui Chau, **Improved Network Robustness with Adversary Critic**. Advances in Neural Information Processing Systems (NIPS), 2018.

Ideally, what confuses neural network should be confusing to humans. However, recent experiments have shown that small, imperceptible perturbations can change the network prediction. To address this gap in perception, we propose a novel approach for learning robust classifier. Our main idea is: adversarial examples for the robust classifier should be indistinguishable from the regular data of the adversarial target. We formulate a problem of learning robust classifier in the framework of Generative Adversarial Networks (GAN), where the adversarial attack on classifier acts as a generator, and the critic network learns to distinguish between regular and adversarial images. The classifier cost is augmented with the objective that its adversarial examples should confuse the adversary critic. To improve the stability of the adversarial mapping, we introduce adversarial cycle-consistency constraint which ensures that the adversarial mapping of the adversarial examples is close to the original. In the experiments, we show the effectiveness of our defense. Our method surpasses in terms of robustness networks trained with adversarial training. Additionally, we verify in the experiments with human annotators on MTurk that adversarial examples are indeed visually confusing.

```txt
@inproceedings{matyasko2018adversarycritic,
    title = {Improved Network Robustness with Adversary Critic},
    author = {Matyasko, Alexander and Chau, Lap-Pui},
    booktitle = {NIPS},
    year = 2018
}
```

## Requirements (tested with python 3.6)
- Tensorflow v1.9.0
- Pytorch (used to save images in the grid)
- Scikit-image

## Training

```bash
python generate_script.py --train=True | bash
```

## Testing

```bash
python generate_script.py --train=False --carlini=False | bash
python generate_script.py --train=False --carlini=True | bash
```
