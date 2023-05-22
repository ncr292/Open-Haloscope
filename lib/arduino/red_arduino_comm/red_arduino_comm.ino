#include <Wire.h>
#include <SPI.h>
#include <Adafruit_BMP280.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

#define BMP_SCK  (13)
#define BMP_MISO (12)
#define BMP_MOSI (11)
#define BMP_CS   (10)

int LightPin = 0;
int BFieldPin = 1;
//float PinToField = 5.0/1024 * 10; // sensitivity of 1mv/G = 10 V / T

//Adafruit_BMP280 bmp; // I2C
Adafruit_BMP280 bmp(BMP_CS, BMP_MOSI, BMP_MISO,  BMP_SCK); // SPI

//Adafruit ADXL345 Accelerometer // SCL, SDA
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);


void setup() {
  pinMode(LED_BUILTIN, OUTPUT);

  Serial.begin(9600);       // initialise UART with baud rate of 9600 bps
  Serial.setTimeout(5);     // timeout to wait all the string to arrive
  Wire.begin();

  bmp.begin();              // initialise the BMP280
  accel.begin();            // initialise the ADXL345
  accel.setRange(ADXL345_RANGE_4_G);
}


void loop() {

  if (Serial.available()) {
    char data_rcvd = Serial.read();   // read one byte from serial buffer and save to data_rcvd

    // get data from accelerometer
    sensors_event_t event; 
    accel.getEvent(&event);

    if (data_rcvd == 'p') Serial.println(bmp.readPressure()); // return pressure
    if (data_rcvd == 't') Serial.println(bmp.readTemperature()); // return temperature
    if (data_rcvd == 'l') Serial.println(analogRead(LightPin)); // return light
    if (data_rcvd == 'b') Serial.println(analogRead(BFieldPin)); // return B-field
  
    if (data_rcvd == 'x') Serial.println(event.acceleration.x); // return acceleration x
    if (data_rcvd == 'y') Serial.println(event.acceleration.y); // return acceleration y
    if (data_rcvd == 'z') Serial.println(event.acceleration.z); // return acceleration z


    digitalWrite(LED_BUILTIN, HIGH); // turn on the led 
    delay(100);
    digitalWrite(LED_BUILTIN, LOW); // turn off the led 
  }

}