function [] = KMeans_TextClassification(distfun,noOfClusters,itrs)
% to load iris data
load sSpamDatabase.dat;
sampleSize = length(sSpamDatabase);

%select random rows
%shuffledArray = sSpamDatabase(randperm(length(sSpamDatabase)),:);
%dlmwrite('before.txt',shuffledArray,'delimiter','\t','precision',4);
%normc(shuffledArray(:,1:57));
%dlmwrite('after.txt',normc(shuffledArray(:,1:57)),'delimiter','\t','precision',4);

%load centers
load centers.dat
cenLen = length(centers);

%center matrixes
meanMat{itrs} = []; 

%confusion matrix
confMat{itrs} = [];
sseItr = [];
precisionRecall = [];

%normalization - 
% for col=1:size(sSpamDatabase,2)-1;
%     minVal = min(sSpamDatabase(:,col));
%     maxVal = max(sSpamDatabase(:,col));
%     sSpamDatabase(:,col) = (sSpamDatabase(:,col) - minVal)/(maxVal-minVal);
% end
% 
% for col=1:size(centers,2)-1;
%     minVal = min(centers(:,col));
%     maxVal = max(centers(:,col));
%     centers(:,col) = (centers(:,col) - minVal)/(maxVal-minVal);
% end


%Clustering
switch upper(distfun)
    case 'EUC'
            spam{itrs} = [];
            nonspam{itrs} = [];
                     for itr = 1:itrs
                                 for ex = 1:sampleSize
                                     distFromClus = ones(1,2);
                                     for cls = 1:noOfClusters
                                         elemDiff = sSpamDatabase(ex,1:57) - centers(cls,1:57);
                                         dist = sqrt(elemDiff * elemDiff');
                                         distFromClus(1,cls) = dist;
                                     end

                                     if distFromClus(1,1) < distFromClus(1,2)
                                         spam{itr} = [spam{itr};sSpamDatabase(ex,:)];
                                     else
                                         nonspam{itr} = [nonspam{itr};sSpamDatabase(ex,:)];
                                     end
                                 end
                           meanMat{itr} = centers;
                           cent1 =  mean(spam{itr}(:,1:57));
                           cent2 =  mean(nonspam{itr}(:,1:57));
                           centers(1,1:57) = cent1;
                           centers(2,1:57) = cent2;
                           
                        %SSE Error Calculation
                        sSSE = 0;
                        nsSSE=0;
                         for i = 1:length(spam{itr})
                                 selemDiff = spam{itr}(i,1:57) - cent1;
                                 sdist = (selemDiff * selemDiff');
                                 sSSE = sSSE+sdist;
                         end
                         for j = 1:length(nonspam{itr})
                                 nselemDiff = nonspam{itr}(j,1:57) - cent2;
                                 nsdist = (nselemDiff * nselemDiff');
                                 nsSSE = nsSSE+nsdist;
                         end
                                     
                       sse = (nsSSE+sSSE)/1.0e+09;
                       vect = [itr real(sse)];
                       sseItr = [sseItr;vect];
                       %calculating confusion matrix
                       TP = 0.0;
                       FP = 0.0;
                       FN = 0.0;
                       TN = 0.0;

                       for i = 1:length(spam{itr})
                           if spam{itr}(i,58) == 1
                               TP = TP+1;
                           else
                               FP = FP+1;
                           end
                       end

                       for i = 1:length(nonspam{itr})
                           if nonspam{itr}(i,58) == 0
                               TN = TN+1;
                           else
                               FN = FN+1;
                           end
                       end
                       confMat{itr} = [TP FN;FP TN];
                       precision = TP/(TP+FP);
                       recall = TP/(TP+FN);
                       accuracy = (TP+TN)/(TP+FN+FP+TN);
                       vec = [itr precision recall accuracy];
                       precisionRecall = [precisionRecall;vec];
                       disp(itr);
                       disp(confMat{itr});
                       X = sprintf('Precision-->%f',precision);
                       Y = sprintf('Recall-->%f',recall);
                       Z = sprintf('Accuracy-->%f',accuracy);
                       disp(X);
                       disp(Y);
                       disp(Z);
                     end    
                     dlmwrite('newCenters.dat',meanMat{itrs},'delimiter','\t','precision',4);
    case 'COSINE'
        spam{itrs} = [];
        nonspam{itrs} = [];
        c = sum(sSpamDatabase~=0,1);
        idff = @(x) log(sampleSize/x);
        tff = @(x) x*100;
        
        V1 = repmat(c(:,1:54),sampleSize,1);
        idf = arrayfun(idff,V1);
        tf = arrayfun(tff,sSpamDatabase(:,1:54));
        wt = times(idf,tf);
        inputdata = cat(2,wt,sSpamDatabase(:,55:58));
        
        V2 = repmat(c(:,1:54),2,1);
        idfc = arrayfun(idff,V2);
        tfc = arrayfun(tff,centers(:,1:54));
        wtc = times(idfc,tfc);
        centroiddata = cat(2,wtc,centers(:,55:58));
        for itr = 1:itrs
            for ex = 1:sampleSize
              distFromClus = ones(1,2);
                  for cls = 1:noOfClusters
                      dist = dot(sSpamDatabase(ex,1:57),centers(cls,1:57))/(norm(sSpamDatabase(ex,1:57))*norm(centers(cls,1:57)));
                      distFromClus(1,cls) = acos(dist);
                  end

                  if distFromClus(1,1) < distFromClus(1,2)
                        spam{itr} = [spam{itr};sSpamDatabase(ex,:)];
                  else
                        nonspam{itr} = [nonspam{itr};sSpamDatabase(ex,:)];
                  end
            end
            meanMat{itr} = centers;
            cent1 =  mean(spam{itr}(:,1:57));
            cent2 =  mean(nonspam{itr}(:,1:57));
            centers(1,1:57) = cent1;
            centers(2,1:57) = cent2;
            
            %SSE Error Calculation
                        sSSE = 0;
                        nsSSE=0;
                         for i = 1:length(spam{itr})
                                 selemDiff = spam{itr}(i,1:57) - cent1;
                                 sdist = (selemDiff * selemDiff');
                                 sSSE = sSSE+sdist;
                         end
                         for j = 1:length(nonspam{itr})
                                 nselemDiff = nonspam{itr}(j,1:57) - cent2;
                                 nsdist = (nselemDiff * nselemDiff');
                                 nsSSE = nsSSE+nsdist;
                         end
                                     
                       sse = (nsSSE+sSSE);
                       rSSE= (sse + conj(sse))/2;
                       vect = [itr rSSE];
                       sseItr = [sseItr;vect];
            %calculating confusion matrix
                       TP = 0.0;
                       FP = 0.0;
                       FN = 0.0;
                       TN = 0.0;

                       for i = 1:length(spam{itr})
                           if spam{itr}(i,58) == 1
                               TP = TP+1;
                           else
                               FP = FP+1;
                           end
                       end

                       for i = 1:length(nonspam{itr})
                           if nonspam{itr}(i,58) == 0
                               TN = TN+1;
                           else
                               FN = FN+1;
                           end
                       end
                       confMat{itr} = [TP FN;FP TN];
                       precision = TP/(TP+FP);
                       recall = TP/(TP+FN);
                       accuracy = (TP+TN)/(TP+FN+FP+TN);
                       vec = [itr precision recall accuracy];
                       precisionRecall = [precisionRecall;vec];
                       disp(confMat{itr});
                       X = sprintf('Precision-->%f',precision);
                       Y = sprintf('Recall-->%f',recall);
                       Z = sprintf('Accuracy-->%f',accuracy);
                       disp(X);
                       disp(Y);
                       disp(Z);
        end
        dlmwrite('newCenters.dat',meanMat{itrs},'delimiter','\t','precision',4);
end  
       
    plot(sseItr);
    plotyy(precisionRecall(:,1),precisionRecall(:,2),precisionRecall(:,1),precisionRecall(:,3));
    plot(precisionRecall(:,1),precisionRecall(:,4));
    
 end