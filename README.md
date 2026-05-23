# Project Instructions

In this project, students are asked to perform the following objectives:

1) DICOM loading and visualization

    a) Download the sample dynamic PET study e 1 BRAIN DINAMIC COLINA, and the associated MR image AX 3D T1 provided by the professor.

    b) Visualize both images with the help of a third party DICOM visualizer that supports dynamic PET imaging.

    c) Load the dynamic PET study. Rearrange the image’s ‘pixel array’ given by PyDicom based on the headers. Some relevant headers include:

        - (0028, 0008) Number of Frames, (0028, 0010) Rows, (0028, 0011) Columns
        - (0018, 0088) Spacing Between Slices, (0028, 0030) Pixel Spacing
        - (0055, 1002) [Frame Positions Vector]
        - (0055, 1001) [Frame Start Times Vector], (0055, 1004) [Frame Durations (ms) Vector]
        
    d) Visualize the last frame, and the average of all the frames.

    e) Create an animation (e.g. gif ﬁle) to display the 3 median planes across the diﬀerent frames.

2) 3D Rigid Coregistration

    a) Coregister the average of all frames (input) to the MR image (reference). Tou can either implementing all steps of the image coregistration yourself, or consider external libraries such as PyElastix.

    b) Create an animation (e.g. gif ﬁle) to display a rotating Maximum Intensity Projection on the coronal-sagittal planes, of:

        i) the reference image, 

        ii) the co-registered input image, and

        iii) an alpha-fusion of both of them.

3) 3D Image Segmentation

    a) Consider the tumor which is clearly visible in the last frame of the PET image. Manually ﬁnd its approximate center and bounding box.

    b) Use an AI general-purpuse segmentation model (e.g. nnInteractive, SAMed-2, SAT, Med-SAM2, etc.) to obtain a semi-automatic tumor segmentation algorithm that only uses the MR image, and either the bounding box, the centroid of the tumor or a textual prompt.

    c) Visualize the segmented Tumor mask on the image. Assess the correctness of the algorithm, numerically and visually.