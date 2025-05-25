clear all;
close all;
clc;

% Configuração inicial HOUVE UM ENGANO PQ A GENTE PENSOU QUE NAO PODIA SER
% DADO DESSA FORMA MAS FOI CORRIGIDO NA DEFESA
% Usar caminho absoluto
    current_dir = pwd;
    pasta_start = fullfile(current_dir, 'start');
    
    classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};  % Nomes exatos das pastas
    num_classes = length(classes);
    imagens_por_classe = 5;
    
    % Carregar e processar imagens
    [inputs, targets] = carregar_imagens(pasta_start, classes, imagens_por_classe);
    
    mediaAccuracy = 0;
    resultsDir = 'resultados_alinea_a\';
    topologia = 10;
    trainFunc = 'trainlm';

    for i = 0:10
    net = feedforwardnet(topologia); % Uma camada de 10 neuronios
    
    % Funçao de treino
    %net.trainFcn = trainFunc;
  
    % Configurar funções de ativação
    %net.layers{1}.transferFcn = 'logsig';
    %net.layers{2}.transferFcn = 'purelin';
     
    % Divisão
    % net.divideFcn = 'dividerand';
    % net.divideParam.trainRatio = 0.70;
    % net.divideParam.valRatio = 0.15;
    % net.divideParam.testRatio = 0.15;
    
    % Treinar rede
        [net, tr] = train(net, inputs, targets);
        out = sim(net, inputs);
            %erro = perform(net, out,in);
            %Calcula e mostra a percentagem de classificacoes corretas no total dos exemplos
            r=0;
            for j=1:size(out,2)               % Para cada classificacao  
                [a , b] = max(out(:,j));          %b guarda a linha onde encontrou valor mais alto da saida obtida
                [c , d] = max(targets(:,j));              %d guarda a linha onde encontrou valor mais alto da saida desejada
              if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
                  r = r+1;
              end
            end
            accuracy = r/size(out,2)*100;
            fprintf('Precisao global %f\n', accuracy)
        mediaAccuracy = mediaAccuracy + accuracy;
        fprintf('media de precisao global %f\n', mediaAccuracy/10);
        executionData = struct();
        executionData.network = net;       % Trained network
        executionData.trainingRecord = tr; % Training metrics
        executionData.parameters = struct(...
            'topology', topologia, ...
            'activation', layers, ...
            'trainFcn', trainFunc);
        executionData.performance = struct(...
            'trainPrecision', accuracy);
        filename = fullfile(resultsDir, ['execution_dataTopologias_', num2str(i), '.mat']);
        disp(1);
        save(filename, '-struct', 'executionData'); 
        %writetable(struct2table(executionData.performance), ...
        %fullfile(resultsDir, sprintf('execution_dataConf2_%d.mat', i)));
    end
    



% Função para carregar e processar imagens
function [X, Y] = carregar_imagens(pasta, classes, imagens_por_classe)
    X = [];
    Y = [];
    
    if ~exist(pasta, 'dir')
        error('A pasta %s não existe!', pasta);
    end
    
    for i = 1:length(classes)
        pasta_classe = fullfile(pasta, classes{i});
    
        
        if ~exist(pasta_classe, 'dir')
            fprintf('AVISO: A pasta %s não existe!\n', pasta_classe);
            continue;
        end
       
        arquivos = dir(fullfile(pasta_classe, '*.png'));
      
       disp(arquivos)
        
        for j = 1:min(imagens_por_classe, length(arquivos))
            % Carregar imagem
            nome_arquivo = fullfile(pasta_classe, arquivos(j).name);
            
            try
                img = imread(nome_arquivo);
          
                
                % Converter para binário e redimensionar
                img_bin = imbinarize(img);
                img_resized = imresize(img_bin, [28 28]);
                
                
                % Converter para vetor
                X = [X, double(img_resized(:))];
                
                % Criar vetor de saída one-hot
                y = zeros(length(classes), 1);
                y(i) = 1;
                Y = [Y, y];
                
               
            catch e
                fprintf('ERRO ao processar arquivo %s: %s\n', nome_arquivo, e.message);
                continue;
            end
        end
    end
    
    if isempty(X)
        error('Nenhuma imagem foi carregada. Verifique se o caminho está correto e se existem imagens nas pastas.');
    end
end 