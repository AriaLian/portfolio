+++
title = "Local Binary Patterns (LBP) for Texture Classification"
summary = "Using the LBP heuristics to distinguish some pattern groups."
description = ""
featuredImage = ""
tags = ["LBP"]
categories = ["AI"]
collections = [""]
weight = 5
draft = false
+++

There are lots of different types of texture descriptors that are used to extract features of an image. Local Binary Pattern, also known as LBP, is a simple and grayscale invariant texture descriptor measure for classification. 

In LBP, a binary code is generated at each pixel by thresholding it’s neighbourhood pixels to either 0 or 1 based on the value of the centre pixel. In this project, I used the LBP heuristics to distinguish some pattern groups.

{{< button href="https://colab.research.google.com/drive/1o2rAij_WTOS1YNKjADXOa5odWk9BqDsg" target="_self" >}}
{{< icon "link" >}} View on Google Colab
{{< /button >}}

![](00.jpeg)

