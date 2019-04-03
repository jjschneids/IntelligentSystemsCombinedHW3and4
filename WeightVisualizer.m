% modified Weight Viewer (Dr. Minai)


figure;
U = Weights;


for i=1:10 %20
    for j = 1:10
        v = reshape(U{2,1}(j + (i-1)*10,:),28,28);
        subplot(10,10,(i-1)*10+j)%subplot(10,20,(i-1)*10+j)
        image(64*v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end