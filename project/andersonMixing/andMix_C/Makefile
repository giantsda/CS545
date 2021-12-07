gaussj.o: gaussj.c  
	gcc gaussj.c  -c -g -Wall

chen_util.o: chen_util.c
	gcc chen_util.c -c -g -Wall

ADM_chen_C.o: ADM_chen_C.c
	gcc ADM_chen_C.c -c -g -Wall

nrutil.o: nrutil.c
	gcc nrutil.c -c -g -Wall

driveADM.o: driveADM.c   nrutil.o
	gcc driveADM.c nrutil.o -c -g -Wall

driveADM: driveADM.o gaussj.o ADM_chen_C.o nrutil.o chen_util.o 
	gcc nrutil.o driveADM.o gaussj.o ADM_chen_C.o chen_util.o -o driveADM -Wall


.PHONY: all

all:driveADM


.PHONY: run

run:  
	make all
	./driveADM
	 
.PHONY: clean

clean :
	rm *.o driveADM  
 
