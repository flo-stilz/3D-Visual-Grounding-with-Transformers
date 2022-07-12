# 3D-Visual-Grounding-with-Transformers
(add qualitative results as a impression)


## Introduction
3D visual grounding is the task of localizing a target object in a 3D scene given a natural language description. This work focuses on developing a transformer architecture for bounding box prediction around a target object that is described by a natural language description.

For additional detail, please see the ScanRefer paper:  
"[3D Visual Grounding with Transformers](https://arxiv.org/abs/1912.08830)"  
by [Stefan Frisch](https://github.com/ga92xug)) and [Florian Stilz](https://github.com/flo-stilz/))
from [Technical University of Munich](https://www.tum.de/en/).

## Setup + Dataset
For the setup and dataset preparation please check the ScanRefer [github page](https://github.com/daveredrum/ScanRefer).

## Results
To reproduce our results we provide the following commands along with the results.

<table>
    <col>
    <col>
    <colgroup span="2"></colgroup>
    <col>
    <tr>
        <th rowspan=2>Name</th>
        <th rowspan=2>Command</th>
        <th colspan=2 scope="colgroup">Overall</th>
        <th rowspan=2>Comments</th>
    </tr>
    <tr>
        <td>Acc<!-- -->@<!-- -->0.25IoU</td>
        <td>Acc<!-- -->@<!-- -->0.5IoU</td>
    </tr>
    <tr>
        <td>ScanRefer (Baseline)</td>
        <td><pre lang="shell">python script/train.py 
        --use_color --use_chunking</pre></td>
        <td>37.05</td>
        <td>23.93</td>
        <td>rgb + color + height</td>
    </tr>
    <tr>
        <td>Ours (pretrained 3DETR-m + GRU + vTransformer) </td>
        <td><pre lang="shell">python script/train.py 
        --use_color --use_chunking --detection_module detr 
        --match_module transformer</pre></td>
        <td>37.08</td>
        <td>26.34</td>
        <td>rgb + color + height</td>
    </tr>

</table>

