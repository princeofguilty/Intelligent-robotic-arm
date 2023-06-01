clear ard;
ard=arduino('COM6');
servoAttach(ard,2);
servoAttach(ard,3);
servoAttach(ard,4);
servoAttach(ard,5);
L(1)=Link([0 5.3 0 pi/2]);
L(2)=Link([0 0 7.7 pi]);
L(3)=Link([0 0 10 0]);
Rob=SerialLink(L);
theta1=0;
theta2=90;
theta3=90;
theta4=180;
openGripper=90;
closeGripper=180;
servoWrite(ard,2,theta1);
servoWrite(ard,3,180-theta2);
servoWrite(ard,4,180-theta3);
servoWrite(ard,5,theta4);
pause(3);
Rob.plot([theta1*pi/180 theta2*pi/180 theta3*pi/180]);
class=[0 0 0 0;
        0 0 0 0;
        0 0 0 0];
input_socket=-1;
while input_socket==-1
    input_socket=socketInit('172.20.10.13',8000,-1);
end
while true
    msg=getmsg(input_socket);
    splitLines=splitlines(msg);
    if (length(splitLines)<3)
        continue;
    end
    for i= 1:length(splitLines)-1
        splitSpace=split(splitLines(i));
        for j=1:4
            class(str2double(splitSpace(1))+1,j)=str2double(splitSpace(j+1));
        end
    end
    disp(class);
    socClose(input_socket);
    break;
end

d=6;
servoWrite(ard,5,openGripper);
pause(3);

if (class(1,3)>class(1,4))
    xbg=(class(2,1)-class(1,1))*d/class(1,3);
    xbo=(class(3,1)-class(1,1))*d/class(1,3);
    ybg=(class(2,2)-class(1,2))*d/class(1,3);
    ybo=(class(3,2)-class(1,2))*d/class(1,3);
else
    xbg=(class(2,1)-class(1,1))*d/class(1,4);
    xbo=(class(3,1)-class(1,1))*d/class(1,4);
    ybg=(class(2,2)-class(1,2))*d/class(1,4);
    ybo=(class(3,2)-class(1,2))*d/class(1,4);
end

a=sqrt(xbg^2+ybg^2);
c=sqrt(xbo^2+ybo^2);
b=sqrt((xbo-xbg)^2+(ybo-ybg)^2);

theta=acos((a^2+b^2-c^2)/(2*a*b));
posx=a-b*cos(theta);
posy=b*sin(theta);
T=[];
T=[ 1 0 0 posx;
    0 1 0 posy;
    0 0 1 3;
    0 0 0 1];
J=[];
J=Rob.ikine(T,'mask',[1 1 1 0 0 0])*180/pi;

theta1=J(1);
theta2=J(2);
theta3=J(3);
if theta2<0 && theta3<0
    theta2=abs(theta2);
    theta3=abs(theta3);
end
Rob.plot([theta1*pi/180 theta2*pi/180 theta3*pi/180]);
servoWrite(ard,2,round(theta1));
pause(3)
servoWrite(ard,4,round(180-theta3)); 
pause(3);
servoWrite(ard,3,round(180-theta2));
pause(3);
servoWrite(ard,5,closeGripper);
pause(3);
theta1=0;
theta2=90;
theta3=90;
theta4=180;
servoWrite(ard,3,180-theta2);
servoWrite(ard,4,180-theta3);
pause(3);
servoWrite(ard,2,theta1);
pause(3);
servoWrite(ard,5,openGripper);
