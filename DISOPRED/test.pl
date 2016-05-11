#!/usr/bin/perl -w
use Cwd 'abs_path';
use File::Basename;
print dirname(abs_path($0))."\n";


