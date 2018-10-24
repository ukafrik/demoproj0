
I'm using Tess4J and it was the ghost4J module within that which was throwing the exception while converting a PDF to an image for OCR, but it turned out not to be ghost4J that was the problem.

The issue seems to be that Tess4J comes bundled with ghost4j 1.0.1 and JNA 4.2.2. If I switch out JNA 4.2.2 for JNA 4.1.0, that seems to fix the "Invalid calling convention" error and allows ghost4j to do the conversion without issue.

For reference I am on CentOS 6.4 and GS 8.70.
