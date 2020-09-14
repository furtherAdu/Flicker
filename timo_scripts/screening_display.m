clear all;

eeglab;

%EEG=pop_readbdf('Timo_Test.bdf');
EEG=pop_readbdf('HW20.bdf');


% pop_saveset(tt,'filename','Timo_Test.set');

delchan = 1:EEG.nbchan;

%finding the index of the Erg1 chanel
Erg1 = find(strcmp({EEG.chanlocs.labels}, 'Erg1')==1);
% 19/20 electrodes selection
oichan  = [1 34 7 5 38 40 42 15 13 48 50 52 23 21 57 58 60 27 64 Erg1]; %[1 34 7 5 38 40 42 15 13 48 50 52 23 21 57 58 60 27 64 Erg1];
delchan(oichan) = [];
% delete chanels that are not selected
nbchan=EEG.nbchan;
EEG.data(delchan,:)=[];
EEG.chanlocs(delchan)=[];
EEG.nbchan=nbchan-length(delchan);

%filtering
Wn=[1 45]/EEG.srate*2; %Bandpass breite ()
[b,a]=butter(3,Wn); %Butterworthfilter 3. Ordnung, Koeffizientenberechnung
EEG.data(1:end-1,:)=filtfilt(b,a,double(EEG.data(1:end-1,:))')';    %filter


%resample
EEG = pop_resample( EEG, 256);


for i =1:size(EEG.data,1)
    EEG.data(i,:) = EEG.data(i,:) - mean(EEG.data(i,:));
end

colarr={};
for i=1:EEG.nbchan
    colarr(i)={[0.4 0.4 0.5]};
end
colarr(end)={[1 0 0]};


EEG.data(end,:) = EEG.data(end,:)>0;
EEG.data(end,:) = EEG.data(end,:)*100;


pop_eegplot(EEG,1,1,1,0,'color',colarr);
