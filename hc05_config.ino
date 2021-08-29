#include <SoftwareSerial.h>
    
SoftwareSerial mySerial(10, 11); // RX, TX  
String command = ""; 

void setup()   
{  
 
  Serial.begin(115200);  
  Serial.println("Type AT commands:");  

  mySerial.begin(38400);  
}  
    
void loop()  
{  
   
  if (mySerial.available()) 
  {  
     while(mySerial.available()) 
     {
       command += (char)mySerial.read();  
     }  
   Serial.println(command);  
   command = ""; 
  }  

  if (Serial.available())
  {  
    delay(10); 
    mySerial.write(Serial.read());  
  }  
}
