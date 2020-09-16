<?php

require_once($_SERVER['DOCUMENT_ROOT'] . 'vendor/autoload.php');

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\Metrics\SMAPE;

$testingData = new CSV(__DIR__.'\datasets\testing.csv', true);
$testingDataset = Labeled::fromIterator($testingData) -> transformLabels('floatval');

$persister = new Filesystem(__DIR__.'\output\estimator.model');
$estimator = $persister -> load();

echo "Rodando testes..." . PHP_EOL;
$results = $estimator -> predict($testingDataset);

$metric = new SMAPE();

$score = $metric -> score($results, $testingDataset -> labels());
$percentage = number_format( (float)$score * -1 , 2, '.', ''); 

echo 'O modelo alcançou uma precisão de ' . $percentage . '%';
