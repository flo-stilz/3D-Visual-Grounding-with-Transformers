# 3D-Visual-Grounding-with-Transformers


## Introduction
3D visual grounding is the task of localizing a target object in a 3D scene given a natural language description. This work focuses on developing a transformer architecture for bounding box prediction around a target object that is described by a natural language description.

## Setup + Dataset +
For the setup and dataset preparation please check the ScanRefer [github page](https://github.com/daveredrum/ScanRefer).

## Models
To reproduce our results we provide the following model along with the weights and the command.

<table>
    <col>
    <col>
    <colgroup span="2"></colgroup>
    <col>
    <tr>
        <th rowspan=2>Name</th>
        <th rowspan=2>Command</th>
        <th colspan=2 scope="colgroup">Overall</th>
        <th rowspan=2>Weights</th>
    </tr>
    <tr>
        <td>Acc<!-- -->@<!-- -->0.25IoU</td>
        <td>Acc<!-- -->@<!-- -->0.5IoU</td>
    </tr>
    <tr>
        <td>xyz</td>
        <td><pre lang="shell">python script/train.py --no_lang_cls</pre></td>
        <td>36.01</td>
        <td>23.76</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ.zip>weights</a></td>
    </tr>

</table>

