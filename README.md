# Open Haloscope
 An open source axion dark matter experiment, for real.
 
 Axions are an hypothetical particles which extend the Standard Model of particle physics and, if they exist, form dark matter.
 The search for axions is performed with earth-based experiments called haloscopes from the dark matter halo of the Milky Way.
 To build an haloscope, one normally needs a partcile physics laboratory, a team of physicists and engineers, and a few million euros.
 
 This project implements a fermionic axion interferometer based on the axion-spin interaction. Conversely to usual haloscopes, it can be built by anybody, in a garage, and at a cost of roughly 250€. A complete experimental setup allows for characterisation measurements, data acquisition and processing, data analysis, and eventually leads to investigating the existance of dark matter axions. In case this is not enough, this same experiment is most likely sensitive to gravitational waves as well.
 
 For more detailed information, building instructions and so on, please refer to the wiki of the project.
 
 ---
 
 ### The experiment
  The physical phenomenon which is sensitive to the presence of dark matter is a magnetic resonance, and in particular its frequency.
  In a few words, the experiments monitors the MHz-frequency resonances of two perpendicular [ferrimagnetic rods](https://fr.rs-online.com/web/p/noyaux-de-ferrites/4673983), using a [Red Pitaya](https://redpitaya.readthedocs.io/en/latest/index.html) board. In addition, several other physical parameters are measure with the aid of an [Arduino](https://www.arduino.cc/) and some sensors.

![Alt text](lib/open-haloscope.png?raw=true "Title")
