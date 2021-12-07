gaussj.o: gaussj.cpp  
	g++ gaussj.cpp  -c -g -Wall

ADM_Chen.o: ADM_Chen.cpp
	g++ ADM_Chen.cpp -c -g -Wall

driveADM.o: driveADM.cpp    
	g++ driveADM.cpp -c -g -Wall

driveADM: driveADM.o gaussj.o ADM_Chen.o   
	g++ driveADM.o gaussj.o ADM_Chen.o -o driveADM -Wall


.PHONY: all

all:driveADM


.PHONY: run

run:  
	make all
	./driveADM
	 
.PHONY: clean

clean :
	rm *.o driveADM  
 
