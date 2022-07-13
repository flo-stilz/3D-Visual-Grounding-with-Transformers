# 3D Visual Grounding with Transformers

<p align="center"><img src="paper & figures/qual_analysis2.jpg" width="800px"/></p>

## Introduction
3D visual grounding is the task of localizing a target object in a 3D scene given a natural language description. This work focuses on developing a transformer architecture for bounding box prediction around a target object that is described by a natural language description.

For additional detail, please see our paper:  
"[3D Visual Grounding with Transformers](https://github.com/flo-stilz/3D-Visual-Grounding-with-Transformers/blob/lang-det/paper%20%26%20figures/final_submission_3D_visual_grounding_with_transformers.pdf)"  
by [Stefan Frisch](https://github.com/ga92xug) and [Florian Stilz](https://github.com/flo-stilz/)
from the [Technical University of Munich](https://www.tum.de/en/). 

## Setup + Dataset
For the setup and dataset preparation please check the ScanRefer [github page](https://github.com/daveredrum/ScanRefer).

## Architecture
In our architecture we replaced VoteNet by 3DETR-m and added a vanilla transformer encoder to the fusion module.
<p align="center"><img src="paper & figures/Ours_final.drawio.png" width="1000px"/></p>


## Results
To reproduce our results we provide the following commands along with the results. The pretrained models are in the outputs folder.
We have implemented a chunking mechanism which significantly reduced the training time compared to the normal ScanRefer. The training of the baseline model takes around 4 hours on a current GPU (NVIDIA Tesla T4).

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
        --use_color --lr 1e-3 --batch_size 14</pre></td>
        <td>37.05</td>
        <td>23.93</td>
        <td>rgb + color + height</td>
    </tr>
    <tr>
        <td>ScanRefer with pretrained VoteNet (optimized Baseline)</td>
        <td><pre lang="shell">python script/train.py 
        --use_color --use_chunking 
        --use_pretrained 'folder' 
        --lr 1e-3 --batch_size 14</pre></td>
        <td>37.11</td>
        <td>25.21</td>
        <td>rgb + color + height</td>
    </tr>
    <tr>
        <td>Ours (pretrained 3DETR-m + GRU + vTransformer) </td>
        <td><pre lang="shell">python script/train.py 
        --use_color --use_chunking 
        --detection_module 3detr 
        --match_module transformer
        --use_pretrained 'folder' </pre></td>
        <td>37.08</td>
        <td>26.34</td>
        <td>rgb + color + height</td>
    </tr>

</table>

