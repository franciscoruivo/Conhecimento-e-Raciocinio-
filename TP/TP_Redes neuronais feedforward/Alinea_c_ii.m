% ========================================================================
% Alínea c) ii) - Treino e teste das melhores redes usando apenas imagens test
% e avaliação com todos os conjuntos
% ========================================================================

clc;
clear all;
close all;

% Carregar funções de suporte (carregarImagens.m deve estar no mesmo diretório)
addpath('.');

% Carregar as melhores redes da alínea b)
fprintf('A carregar as melhores redes da alínea b)...\n');
load('Melhores Redes/melhor_rede_1.mat'); % Rede 1 (divisão 80/10/10)


% Definir parâmetros das redes
rede_params = struct();
rede_params(1).nome = 'Rede 1 (divisão 80/10/10)';
rede_params(1).rede = net;
rede_params(1).divisao = [0.8 0.1 0.1]; % treino/validação/teste
rede_params(1).funcao_treino = 'trainscg';
rede_params(1).topologia = [30];
rede_params(1).epocas = 1000;
load('Melhores Redes/melhor_rede_2.mat'); % Rede 2 (algoritmo trainlm)
rede_params(2).nome = 'Rede 2 (algoritmo trainlm)';
rede_params(2).rede = net;
rede_params(2).divisao = [0.7 0.15 0.15];
rede_params(2).funcao_treino = 'trainlm';
rede_params(2).topologia = [20];
rede_params(2).epocas = 1000;

rede_params(3).nome = 'Rede 3 (topologia [10 10])';
load('Melhores Redes/melhor_rede_3.mat'); % Rede 3 (topologia [10 10])

rede_params(3).rede = net;
rede_params(3).divisao = [0.7 0.15 0.15];
rede_params(3).funcao_treino = 'trainscg';
rede_params(3).topologia = [10 10];
rede_params(3).epocas = 1000;

% Definir classes
classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};
num_classes = length(classes);

% Criar diretório para resultados se não existir
if ~exist('resultados_alinea_c_ii', 'dir')
    mkdir('resultados_alinea_c_ii');
end

% Carregar imagens das três pastas
fprintf('A carregar imagens...\n');
[imagens_start, targets_start] = carregarImagens('start');
[imagens_train, targets_train] = carregarImagens('train');
[imagens_test, targets_test] = carregarImagens('test');

% Criar tabela para armazenar resultados
resultados = table('Size', [9, 8], ...
                  'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'double', 'double'}, ...
                  'VariableNames', {'Rede_Conjunto', 'Precisao_Global', 'Precisao_circle', 'Precisao_kite', ...
                                    'Precisao_parallelogram', 'Precisao_square', 'Precisao_trapezoid', 'Precisao_triangle'});

% Treinar as três redes usando apenas imagens test
for i = 1:length(rede_params)
    fprintf('\n==== Treino da %s com imagens test ====\n', rede_params(i).nome);
    
    % Criar nova rede com a mesma configuração
    hiddenLayerSize = rede_params(i).topologia;
    net = patternnet(hiddenLayerSize, rede_params(i).funcao_treino);
    
    % Configurar divisão de dados
    net.divideParam.trainRatio = rede_params(i).divisao(1);
    net.divideParam.valRatio = rede_params(i).divisao(2);
    net.divideParam.testRatio = rede_params(i).divisao(3);
    
    % Configurar parâmetros de treino
    net.trainParam.epochs = rede_params(i).epocas;
    net.trainParam.showWindow = true;
    
    % Treinar rede usando apenas imagens test
    fprintf('A treinar a rede %d...\n', i);
    [net, tr] = train(net, imagens_test, targets_test);
    
    % Guardar rede treinada
    eval(['rede_' num2str(i) '_test = net;']);
    save(['resultados_alinea_c_ii/rede_' num2str(i) '_test.mat'], ['rede_' num2str(i) '_test']);
    
    % Testar com cada conjunto e guardar resultados
    % Teste com imagens start
    y_start = net(imagens_start);
    [~, cstart] = max(y_start);
    [~, tstart] = max(targets_start);
    precisao_start = 100 * sum(cstart == tstart) / length(tstart);
    
    % Teste com imagens train
    y_train = net(imagens_train);
    [~, ctrain] = max(y_train);
    [~, ttrain] = max(targets_train);
    precisao_train = 100 * sum(ctrain == ttrain) / length(ttrain);
    
    % Teste com imagens test
    y_test = net(imagens_test);
    [~, ctest] = max(y_test);
    [~, ttest] = max(targets_test);
    precisao_test = 100 * sum(ctest == ttest) / length(ttest);
    
    % Guardar precisões na tabela
    idx_start = (i-1)*3 + 1;
    idx_train = (i-1)*3 + 2;
    idx_test = (i-1)*3 + 3;
    
    resultados.Rede_Conjunto(idx_start) = sprintf("Rede %d: start", i);
    resultados.Rede_Conjunto(idx_train) = sprintf("Rede %d: train", i);
    resultados.Rede_Conjunto(idx_test) = sprintf("Rede %d: test", i);
    
    resultados.Precisao_Global(idx_start) = precisao_start;
    resultados.Precisao_Global(idx_train) = precisao_train;
    resultados.Precisao_Global(idx_test) = precisao_test;
    
    % Calcular e guardar matrizes de confusão
    matriz_conf_start = calcularMatrizConfusao(targets_start, y_start);
    matriz_conf_train = calcularMatrizConfusao(targets_train, y_train);
    matriz_conf_test = calcularMatrizConfusao(targets_test, y_test);
    
    % Guardar matrizes de confusão
    save(['resultados_alinea_c_ii/matriz_conf_rede_' num2str(i) '_start.mat'], 'matriz_conf_start');
    save(['resultados_alinea_c_ii/matriz_conf_rede_' num2str(i) '_train.mat'], 'matriz_conf_train');
    save(['resultados_alinea_c_ii/matriz_conf_rede_' num2str(i) '_test.mat'], 'matriz_conf_test');
    
    % Guardar precisões por classe
    for c = 1:num_classes
        resultados{idx_start, c+2} = matriz_conf_start(c,c);
        resultados{idx_train, c+2} = matriz_conf_train(c,c);
        resultados{idx_test, c+2} = matriz_conf_test(c,c);
    end
    
    % Guardar matrizes de confusão em formato texto para visualização
    escreverMatrizConfusao(['resultados_alinea_c_ii/matriz_conf_rede_' num2str(i) '_start.txt'], ...
                          matriz_conf_start, classes, rede_params(i).nome, 'start');
    escreverMatrizConfusao(['resultados_alinea_c_ii/matriz_conf_rede_' num2str(i) '_train.txt'], ...
                          matriz_conf_train, classes, rede_params(i).nome, 'train');
    escreverMatrizConfusao(['resultados_alinea_c_ii/matriz_conf_rede_' num2str(i) '_test.txt'], ...
                          matriz_conf_test, classes, rede_params(i).nome, 'test');
end

% Guardar resultados em Excel
writetable(resultados, 'resultados_alinea_c_ii/resultados_c_ii.xlsx');

% Exibir resultados
disp('Resultados:');
disp(resultados);

fprintf('\nTreino e teste concluídos com sucesso!\n');
fprintf('Os resultados foram guardados na pasta resultados_alinea_c_ii\n'); 