// GRID WORLD MODEL OF A SEMIAUTONOMOUS EXPLORING ROBOT
// Sebastian Junges, RWTH Aachen University
// As described in
// Junges, Jansen, Dehnert, Topcu, Katoen:
// Safety Constrained Reinforcement Learning
// Proc. of TACAS’16

mdp

//PARAMETERS
//The difference of the reliability of the channels between the worst and at the best position
const double pLDiff=0.9;
const double pHDiff=0.1;
//Scaling factor for the minimum reliability of the channels
const double pL;//=8/9;
const double pH;//=1;

//CONSTANTS
//The minimum reliablities
const double pLMin=pL*(1-pLDiff);
const double pHMin=pH*(1-pHDiff);

// Grid size
const int Xsize;
const int Ysize;
// Number of tries before an error
const int MAXTRIES;
// Ball within the robot has to move.
const int B;


formula T = (xLoc = Xsize & yLoc = Ysize);


module robot
  xLoc : [1..Ysize] init 1;
  yLoc : [1..Xsize] init 1;
  unreported : [0..B] init 0;
  hasSendNow : bool init false;
  tries : [0..MAXTRIES] init 0;

  [up] xLoc < Xsize & !T  & hasSendNow  -> 1:(xLoc'=xLoc+1) & (unreported' = 0) & (hasSendNow'=false);
  [up] xLoc < Xsize & !T  & !hasSendNow -> 1:(xLoc'=xLoc+1) & (unreported'=min(unreported+1, B));
  [right] yLoc < Ysize & !T  & hasSendNow  -> 1:(yLoc'=yLoc+1) & (unreported' = 0)& (hasSendNow'=false);
  [right] yLoc < Ysize & !T  & !hasSendNow -> 1:(yLoc'=yLoc+1) & (unreported'=min(unreported+1,B));
  [sendL] !hasSendNow & !T & tries < MAXTRIES -> (pLMin + pLDiff * xLoc/Xsize):(hasSendNow'=true) & (tries'=0) + (1 - pLMin - pLDiff * xLoc/Xsize): (tries'=tries+1);
  [sendH] !hasSendNow & !T & tries < MAXTRIES -> (pHMin + pHDiff * yLoc/Ysize):(hasSendNow'=true) & (tries'=0) + (1 - pHMin - pHDiff * yLoc/Ysize): (tries'=tries+1);
  [done] T -> 1:true;
endmodule

rewards "sendbased"
  [up] true: 0.03;
  [right] true: 0.03;
	[sendL] true: max(10, min(11 + xLoc - yLoc, 20));
	[sendH] true: min(13 + xLoc + yLoc, 24);
endrewards

rewards "sendbased_lower"
  [up] true: 0.03;
  [right] true: 0.03;
	[sendL] true: 10;
	[sendH] true: 12;
endrewards

rewards "sendbased_upper"
  [up] true: 0.03;
  [right] true: 0.03;
	[sendL] true: 20;
	[sendH] true: 24;
endrewards

label "Target" = T;
label "Crash" = unreported=B;
