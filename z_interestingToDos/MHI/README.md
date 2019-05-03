Python version - 3

File Structure - 
Folder - MHI

Folder:
MHI/inputVideo: includes all 6 training video (person15_*.avi) and 6 action combined validation video (testVideo.avi).
MHI/outImage: all output images.
MHI/outputVideo: output video, MHI classification on 6 action combined video (outputTestVideo.avi).
MHI/validationVideo: initial 6 videos from website used to create the combined validation video. 

Files:
MHI/cv_proj.yml: Final project enviornment file (downloaded from Canvas)
MHI/Final_Project_Activity_Classification_using_MHI.docx: Project Requirements
MHI/MHI.docx: word version report
MHI/project.py: execution code
MHI/README.md: nothing but read me:)


Execution code - project.py

line 586: Instantiate object MHIModel
line 588: training model - using all person15 data in folder /inputVideo
line 560: read in test video and output validation test video - test video in /inputVideo and output will be in /outputVideo


Final Deliveray Check List:
Report - MHI/MHI_report.pdf
Test Video - 
	input video  - MHI/inputVideo/testVideo.avi
	output video - MHI/outputVideo/outputTestVideo.avi
  		     - https://drive.google.com/open?id=11mSVYgXX0uxqMg_fEjVZ3bV5SEz6xEoY
		     - https://www.dropbox.com/s/9eost8putmckse1/outputTestVideo.avi?dl=0


Presentation Video  - MHI/videoPresentation.mp4
		    - https://www.dropbox.com/s/qkdhe95cmygq68a/videoPresentation.mp4?dl=0
	            - https://drive.google.com/open?id=1IFK4S6ls4wPTSSZ8IeW2YDbbOBUOpApV

Presentation Slides - MHI/Activity_Classification_using_MHI_presentation.pptx
