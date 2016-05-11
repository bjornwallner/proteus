/* Neural Network Identification of Protein Disorder */
/* Copyright (C) 1998 David T. Jones - Created : March 1998 */

/* Version 3.0 - Created : April 2010 */

/* Training Module */

/*
 * Description: This program provides an implementation of a neural network
 * containing one hidden layer, which uses the generalized backpropagation
 * delta rule for learning.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#define noSOFTMAX
#define noWTBKTRACK

#define notBALANCED
#define notSINGLESEQ

#include "disord_net.h"

enum
{
    FALSE, TRUE
};

#define MAXSQLEN 8000
#define DBSIZE 10000

#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))
#define SQR(x) ((x)*(x))

void           *calloc(), *malloc();

char           *wtfnm;

int             nwtsum, fwt_to[TOTAL], lwt_to[TOTAL];
float            blrate = ILRATE, lrate = ILRATE, alpha = IALPHA;
float            activation[TOTAL], bias[TOTAL], netinput[TOTAL], *weight[TOTAL];
float            delta[TOTAL], *dweight[TOTAL], *wed[TOTAL], *lastwed[TOTAL];
float            bed[TOTAL], lastbed[TOTAL], bias[TOTAL], dbias[TOTAL], bslope[TOTAL], prevbslope[TOTAL];

float            target[NUM_OUT];

int             restot, nblocks, batchflg = 0, rpropflg = 0;


struct entry
{
    char           *id, *seq, *disord;
    int length;
    float         **profile, aacomp[20];
}
db[DBSIZE];

struct samp
{
    short           en, wp, testflg;
}
               *sampord;

char           *rnames[] = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "???"
};

#define ran0() (random()/(RAND_MAX + 1.0))

/* Generate a small random weight (+/- RWMAX) */
#define rndwt() (2.0 * (ran0() - 0.5) * RWMAX)

/* Generate random int 0..X-1 */
#define rndint(x) ((int)((x)*ran0()))

/* Generate a small 'noise' value (0-0.05) */
#define noise() (0.05*ran0())


enum aacodes
{
    ALA, ARG, ASN, ASP, CYS,
    GLN, GLU, GLY, HIS, ILE,
    LEU, LYS, MET, PHE, PRO,
    SER, THR, TRP, TYR, VAL,
    UNK
};

void            err(char *s)
{
    fprintf(stderr, "%s\n", s);
}

void            fail(char *s)
{
    fprintf(stderr, "%s\n", s);
    exit(1);
}


/* Return current CPU time usage */

#ifdef CLOCKS_PER_SEC
#define CLKRATE (CLOCKS_PER_SEC)
#else
#define CLKRATE 1000000
#endif

double          cpu_time()
{
    static double   runtime = 0.0;
    static unsigned long last_t;
    unsigned long   temp;

    temp = clock() & 0x7fffffff;
    if (temp >= last_t)
	runtime += (double) abs(temp - last_t) / CLKRATE;
    else
	runtime += (double) (0x7fffffff - last_t + (temp + 1)) / CLKRATE;
    last_t = temp;
    return (runtime);
}

void            randomize()
{
    srandom((unsigned) time(NULL));
}

/*
 * Back Propagation Engine - as described by McClelland & Rumelhart /
 * Sejnowski & Rosenberg
 */

/* logistic 'squashing' function (+/- 1.0) */
#define logistic(x) (1.0F / (1.0F + (float)exp(-(x))))

void            compute_output()
{
    int             i, j;
    float            netinp, *tp;

    for (i = NUM_IN; i < NUM_IN + NUM_HID; i++)
    {
	netinp = bias[i];
	tp = weight[i];
	for (j = 0; j < NUM_IN; j++)
	    netinp += activation[j] * tp[j];

	/* Trigger neuron */
	activation[i] = logistic(netinp);
    }

    for (i = NUM_IN + NUM_HID; i < TOTAL; i++)
    {
	netinp = bias[i];
	tp = weight[i];
	for (j = NUM_IN; j < NUM_IN + NUM_HID; j++)
	    netinp += activation[j] * tp[j];

	/* Trigger neuron */
#ifdef SOFTMAX
	activation[i] = netinp;
#else
	activation[i] = logistic(netinp);
#endif
    }

#ifdef SOFTMAX
    omax = activation[TOTAL - NUM_OUT];
    for (i = 1; i < NUM_OUT; i++)
	if (activation[TOTAL - NUM_OUT + i] > omax)
	    omax = activation[TOTAL - NUM_OUT + i];
    
    for (sum = i = 0; i < NUM_OUT; i++)
	sum += exp(activation[TOTAL - NUM_OUT + i] - omax);

    for (i = 0; i < NUM_OUT; i++)
	activation[TOTAL - NUM_OUT + i] = exp(activation[TOTAL - NUM_OUT + i] - omax) / sum;
#endif
}

void            compute_error()
{
    int             i, j;
    float          error[TOTAL], *tp, gradbias;

    for (i = 0; i < TOTAL - NUM_IN - NUM_OUT; i++)
	error[i+NUM_IN] = 0.0F;

    for (i = 0; i<NUM_OUT; i++)
	error[i + TOTAL - NUM_OUT] = target[i] - activation[i + TOTAL - NUM_OUT];

    if (rpropflg)
	gradbias = 0.0F;
    else
	gradbias = 0.1F;
    
    for (i = NUM_IN + NUM_HID; i<TOTAL; i++)
    {
	/* Add small bias to derivatives to speed up gradient descent learning */
	delta[i] = error[i] * (gradbias + activation[i] * (1.0F - activation[i]));

	tp = weight[i];
	for (j = 0; j < NUM_HID; j++)
	    error[j+NUM_IN] += delta[i] * tp[j+NUM_IN];
    }
    
    for (i = NUM_IN; i < NUM_IN + NUM_HID; i++)
    {
	delta[i] = error[i] * activation[i] * (1.0F - activation[i]);
	
	tp = weight[i];
	for (j = 0; j < NUM_IN; j++)
	    error[j] += delta[i] * tp[j];
    }
}

void            compute_wed()
{
    int             i, j;
    float *tp;

    for (i = NUM_IN; i < NUM_IN + NUM_HID; i++)
    {
	tp = wed[i];
	for (j = 0; j < NUM_IN; j++)
	    tp[j] += delta[i] * activation[j];
	bed[i] += delta[i];
    }

    for (; i < TOTAL; i++)
    {
	tp = wed[i];
	for (j = NUM_IN; j < NUM_IN + NUM_HID; j++)
	    tp[j] += delta[i] * activation[j];
	bed[i] += delta[i];
    }

    nwtsum++;
}

/* Initialise weight deltas & saved weds */
void            init_rprop(const float ival)
{
    int             i, j;

    for (i = NUM_IN; i < NUM_IN + NUM_HID; i++)
    {
	for (j = 0; j < NUM_IN; j++)
	{
	    dweight[i][j] = ival;
	    lastwed[i][j] = 0.0F;
	}
	
	dbias[i] = ival;
	lastbed[i] = 0.0F;
    }
    
    for (; i < TOTAL; i++)
    {
	for (j = NUM_IN; j < NUM_IN + NUM_HID; j++)
	{
	    dweight[i][j] = ival;
	    lastwed[i][j] = 0.0F;
	}
	
	dbias[i] = ival;
	lastbed[i] = 0.0F;
    }
}

/* Update weights using Rprop */
void            rprop_update_wts()
{
    int             i, j;
    float dwsign;

    for (i = NUM_IN; i < NUM_IN + NUM_HID; i++)
    {
	for (j = 0; j < NUM_IN; j++)
	{
	    dwsign = lastwed[i][j] * wed[i][j];
	    	    
#ifdef WTBKTRACK
	    if (dwsign < 0.0F)
	    {
		if (lastwed[i][j] > 0.0F)
		    weight[i][j] -= dweight[i][j];
		else if (lastwed[i][j] < 0.0F)
		    weight[i][j] += dweight[i][j];
	    }
#endif
	    if (dwsign > 0.0F)
		dweight[i][j] = MIN(dweight[i][j]*1.2F, 50.0F);
	    else if (dwsign < 0.0F)
		dweight[i][j] *= 0.5F;
	    
	    if (-wed[i][j] > 0.0F)
		weight[i][j] -= dweight[i][j];
	    else if (-wed[i][j] < 0.0F)
		weight[i][j] += dweight[i][j];
	    
	    lastwed[i][j] = wed[i][j];
	    wed[i][j] = 0.0F;
	}
	
	dwsign = lastbed[i] * bed[i];
	
#ifdef WTBKTRACK
	if (dwsign < 0.0F)
	{
	    if (lastbed[i] > 0.0F)
		bias[i] -= dbias[i];
	    else if (lastbed[i] < 0.0F)
		bias[i] += dbias[i];
	}
#endif
	
	if (dwsign > 0.0F)
	    dbias[i] = MIN(dbias[i]*1.2F, 50.0F);
	else if (dwsign < 0.0F)
	    dbias[i] *= 0.5F;
	
	if (-bed[i] > 0.0F)
	    bias[i] -= dbias[i];
	else if (-bed[i] < 0.0F)
	    bias[i] += dbias[i];

	lastbed[i] = bed[i];
	bed[i] = 0.0F;
    }

    for (; i < TOTAL; i++)
    {
	for (j = NUM_IN; j < NUM_IN + NUM_HID; j++)
	{
	    dwsign = lastwed[i][j] * wed[i][j];

#ifdef WTBKTRACK
	    if (dwsign < 0.0F)
	    {
		if (lastwed[i][j] > 0.0F)
		    weight[i][j] -= dweight[i][j];
		else if (lastwed[i][j] < 0.0F)
		    weight[i][j] += dweight[i][j];
	    }
#endif
	    
	    if (dwsign > 0.0F)
		dweight[i][j] = MIN(dweight[i][j]*1.2F, 50.0F);
	    else if (dwsign < 0.0F)
		dweight[i][j] *= 0.5F;
	    
	    if (-wed[i][j] > 0.0F)
		weight[i][j] -= dweight[i][j];
	    else if (-wed[i][j] < 0.0F)
		weight[i][j] += dweight[i][j];
	    
	    lastwed[i][j] = wed[i][j];
	    wed[i][j] = 0.0F;
	}
	
	dwsign = lastbed[i] * bed[i];

#ifdef WTBKTRACK
	if (dwsign < 0.0F)
	{
	    if (lastbed[i] > 0.0F)
		bias[i] -= dbias[i];
	    else if (lastbed[i] < 0.0F)
		bias[i] += dbias[i];
	}
#endif
	
	if (dwsign > 0.0F)
	    dbias[i] = MIN(dbias[i]*1.2F, 50.0F);
	else if (dwsign < 0.0F)
	    dbias[i] *= 0.5F;
	
	if (-bed[i] > 0.0F)
	    bias[i] -= dbias[i];
	else if (-bed[i] < 0.0F)
	    bias[i] += dbias[i];

	lastbed[i] = bed[i];
	bed[i] = 0.0F;
    }

    nwtsum = 0;
}


/* Update weights using steepest-descent BP with momentum */
void            sd_update_wts()
{
    int             i, j;
    float *tp1, *tp2, *tp3, *tp4, lr, blr;

    lr = lrate / nwtsum;
    blr = blrate / nwtsum;

    for (i = NUM_IN; i < NUM_IN + NUM_HID; i++)
    {
	tp1 = dweight[i];
	tp2 = wed[i];
	tp3 = weight[i];
	for (j = 0; j < NUM_IN; j++)
	{
	    tp1[j] = lr * tp2[j] + IALPHA * tp1[j];
	    tp3[j] += tp1[j];
	    tp2[j] = 0.0F;
	}
	dbias[i] = blr * bed[i] + IALPHA * dbias[i];
	bias[i] += dbias[i];
	bed[i] = 0.0F;
    }
    
    for (; i < TOTAL; i++)
    {
	tp1 = dweight[i];
	tp2 = wed[i];
	tp3 = weight[i];
	for (j = NUM_IN; j < NUM_IN + NUM_HID; j++)
	{
	    tp1[j] = lr * tp2[j] + IALPHA * tp1[j];
	    tp3[j] += tp1[j];
	    tp2[j] = 0.0F;
	}
	dbias[i] = blr * bed[i] + IALPHA * dbias[i];
	bias[i] += dbias[i];
	bed[i] = 0.0F;
    }

    nwtsum = 0;
}


/*
 * Return half of the sum of error squares (output versus target)
 */
float            pattern_error()
{
    int             i;
    float            temp, sum = 0.0;

    for (i = 0; i < NUM_OUT; i++)
    {
	temp = target[i] - activation[TOTAL - NUM_OUT + i];
	sum += SQR(temp);
    }

    return (sum * 0.5F);
}

/*
 * save weights - save all link weights to a disk file
 */
void            save_wts(char *fname)
{
    int             i, j;
    FILE           *ofp;

    if (!(ofp = fopen(fname, "w")))
	fail("save_wts: unable to open file");

    /* Save input units to hidden layer weights */
    for (i = NUM_IN; i < NUM_IN + NUM_HID; i++)
	for (j = 0; j < NUM_IN; j++)
	    fprintf(ofp, "%.8g\n", weight[i][j]);

    /* Save hidden layer to output units weights */
    for (; i < TOTAL; i++)
	for (j = NUM_IN; j < NUM_IN + NUM_HID; j++)
	    fprintf(ofp, "%.8g\n", weight[i][j]);

    /* Save bias weights */
    for (j = NUM_IN; j < TOTAL; j++)
	fprintf(ofp, "%.8g\n", bias[j]);

    fclose(ofp);
}

/*
 * load weights - load all link weights from a disk file
 */
void            load_wts(char *fname)
{
    int             i, j;
    double          t;
    FILE           *ifp;

    if (!(ifp = fopen(fname, "r")))
    {
	printf("Creating new file : %s ...\n", fname);
	return;
    }

    /* Load input units to hidden layer weights */
    for (i = NUM_IN; i < NUM_IN + NUM_HID; i++)
	for (j = 0; j < NUM_IN; j++)
	{
	    fscanf(ifp, "%lf", &t);
	    weight[i][j] = t;
	}

    /* Load hidden layer to output units weights */
    for (; i < TOTAL; i++)
	for (j = NUM_IN; j < NUM_IN + NUM_HID; j++)
	{
	    fscanf(ifp, "%lf", &t);
	    weight[i][j] = t;
	}

    /* Load bias weights */
    for (j = NUM_IN; j < TOTAL; j++)
    {
	fscanf(ifp, "%lf", &t);
	bias[j] = t;
    }

    fclose(ifp);
}


/* Initialize network - wire up units and make random weights */
void            init()
{
    int             i, j;

    for (i = NUM_IN; i < TOTAL; i++)
	if (!(weight[i] = calloc(TOTAL - NUM_OUT, sizeof(float))) ||
	    !(dweight[i] = calloc(TOTAL - NUM_OUT, sizeof(float))) ||
	    !(lastwed[i] = calloc(TOTAL - NUM_OUT, sizeof(float))) ||
	    !(wed[i] = calloc(TOTAL - NUM_OUT, sizeof(float))))
	    fail("init: Out of Memory!");

    /* Set first layer weights */
    for (i = NUM_IN; i < NUM_IN + NUM_HID; i++)
	for (j = 0; j < NUM_IN; j++)
	    weight[i][j] = rndwt();

    /* Set second layer weights */
    for (i = NUM_IN + NUM_HID; i < TOTAL; i++)
	for (j = NUM_IN; j < NUM_IN + NUM_HID; j++)
	    weight[i][j] = rndwt();

    /* Randomize bias weights */
    for (j = NUM_IN; j < TOTAL; j++)
	bias[j] = rndwt();
}


/* Convert AA letter to numeric code (0-20) */
int             aanum(int ch)
{
    static int      aacvs[] = {
	999, 0, 20, 4, 3, 6, 13, 7, 8, 9, 20, 11, 10, 12, 2,
	20, 14, 5, 1, 15, 16, 20, 19, 17, 20, 18, 20
    };

    return (isalpha(ch) ? aacvs[ch & 31] : 20);
}

/* Randomly shuffle sampling array */
void            shuffle(struct samp *s, int n)
{
    int             i, ridx;
    struct samp     temp;

    for (i = 0; i < n; i++)
    {
	ridx = rndint(n);
	temp = s[i];
	s[i] = s[ridx];
	s[ridx] = temp;
    }
}

/* Perform one epoch of learning */
void            learn()
{
    int             aa, i, j, k, l, n, winpos, nxmem;
    static short    cpu_upd;

    shuffle(sampord, restot);
    for (n = 0; n < restot; n++)
	if (!sampord[n].testflg)
	{
	    if (!cpu_upd--)
	    {
		cpu_upd = 500;
		(void) cpu_time();	/* Prevent clock wraparound */
	    }

	    winpos = sampord[n].wp;

	    for (j = 0; j < NUM_IN; j++)
		activation[j] = 0.0F;

	    for (j = WINL; j <= WINR; j++)
	    {
		if (j + winpos >= 0 && j + winpos < db[sampord[n].en].length)
		{
		    if (db[sampord[n].en].seq[j + winpos] >= 20)
		    {
			printf("%d %d %d\n", sampord[n].en, j + winpos, db[sampord[n].en].seq[j + winpos]);
			break;
		    }
		    else
			for (aa = 0; aa < 20; aa++)
			    activation[(j - WINL) * IPERGRP + aa] = db[sampord[n].en].profile[j + winpos][aa];
		}
		else
		    activation[(j - WINL) * IPERGRP + 20] = 1.0F;
	    }
	    if (j <= WINR)
		continue;
	    for (aa=0; aa<20; aa++)
		activation[(WINR - WINL + 1) * IPERGRP + aa] = db[sampord[n].en].aacomp[aa];
	    activation[(WINR - WINL + 1) * IPERGRP + 20] = log(1.0 + winpos);
	    activation[(WINR - WINL + 1) * IPERGRP + 21] = log(db[sampord[n].en].length - winpos);
	    target[0] = target[1] = 0.0F;
	    if (db[sampord[n].en].disord[winpos] != '*')
		target[0] = 1.0F;
	    else
		target[1] = 1.0F;

	    compute_output();
	    compute_error();
	    compute_wed();

	    if (!batchflg)
		sd_update_wts();
	}
    
    if (nwtsum)
    {
	if (rpropflg)
	    rprop_update_wts();
	else
	    sd_update_wts();
    }
}

/* Test prediction accuracy on training set */
void            testpred()
{
    int             aa, i, j, k, n, winpos, ncorr, nincorr, fp=0, tp=0, fn=0, tn=0;
    short           pred;
    float            errsum = 0.0, q2, v, mcc;
    static int      nepochs, nincr;
    static float     errmin = 1e32, mccmax = -1;
    static int badprot[10000], nprot[10000];

    for (nincorr = ncorr = n = 0; n < restot; n++)
	if (sampord[n].testflg)
	{
	    winpos = sampord[n].wp;
	    for (j = 0; j < NUM_IN; j++)
		activation[j] = 0.0F;

	    for (j = WINL; j <= WINR; j++)
	    {
		if (j + winpos >= 0 && j + winpos < db[sampord[n].en].length)
		{
		    if (db[sampord[n].en].seq[j + winpos] >= 20)
		    {
//			printf("%d %d %d\n", sampord[n].en, j + winpos, db[sampord[n].en].seq[j + winpos]);
			break;
		    }
		    else
			for (aa = 0; aa < 20; aa++)
			    activation[(j - WINL) * IPERGRP + aa] = db[sampord[n].en].profile[j + winpos][aa];
		}
		else
		    activation[(j - WINL) * IPERGRP + 20] = 1.0F;
	    }
//	    printf("j = %d\n", j);
	    if (j <= WINR)
		continue;
	    for (aa=0; aa<20; aa++)
		activation[(WINR - WINL + 1) * IPERGRP + aa] = db[sampord[n].en].aacomp[aa];
	    activation[(WINR - WINL + 1) * IPERGRP + 20] = log(1.0 + winpos);
	    activation[(WINR - WINL + 1) * IPERGRP + 21] = log(db[sampord[n].en].length - winpos);
	    target[0] = target[1] = 0.0F;
	    if (db[sampord[n].en].disord[winpos] != '*')
		target[0] = 1.0F;
	    else
		target[1] = 1.0F;

	    compute_output();
	    errsum += pattern_error();

	    if (db[sampord[n].en].disord[winpos] == '*' && activation[TOTAL - NUM_OUT] > activation[TOTAL - NUM_OUT + 1])
		badprot[sampord[n].en]++;
	    nprot[sampord[n].en]++;
//	    printf("%d %d %c\n", sampord[n].en, winpos, db[sampord[n].en].disord[winpos]);
	}
    
    for (i=0; i<nblocks; i++)
	printf("%s %f\n", db[i].id, 100.0*badprot[i]/nprot[i]);
}

#define CH malloc_verify(), printf("Heap OK at line : %d.\n",__LINE__);

void           *allocmat(int rows, int columns, int size, int clrflg)
{
    int             i;
    void          **p;

    p = malloc(rows * sizeof(void *));

    if (p == NULL)
	fail("allocmat: malloc [] failed!");
    if (clrflg)
    {
	for (i = 0; i < rows; i++)
	    if ((p[i] = calloc(columns, size)) == NULL)
		fail("allocmat: calloc [][] failed!");
    }
    else
	for (i = 0; i < rows; i++)
	    if ((p[i] = malloc(columns * size)) == NULL)
		fail("allocmat: malloc [][] failed!");

    return p;
}

/* Read in data */
void            read_dat(char *filename)
{
    FILE           *ifp, *tfp;
    char            brkid[10], alnname[512], buf[MAXSQLEN], dsoname[512];
    int             i, j, k, nres, consv[20], nord = 0, ndis = 0, nrep, disof;

    if (!(ifp = fopen(filename, "r")))
	fail("Cannot open training list file!");

    while (!feof(ifp))
    {
	if (fscanf(ifp, "%s", brkid) != 1)
	    break;
	if (brkid[0] == '#')
	    continue;
	strcpy(dsoname, "./data/");
	strcat(dsoname, brkid);
	strcat(dsoname, ".dso");

	puts(dsoname);
	if (!(tfp = fopen(dsoname, "r")))
	    fail("Cannot open DSO file!");
	if (!fgets(buf, 512, tfp))
	    break;
	sscanf(buf, "%*s%*s%d", &nres);
	db[nblocks].id = strdup(brkid);
	db[nblocks].length = nres;
	if (!(db[nblocks].seq = malloc(nres))
	    || !(db[nblocks].disord = malloc(nres)))
	    fail("Out of memory!");
	db[nblocks].profile = allocmat(nres, 20, sizeof(float), FALSE);

	i = 0;
	while (!feof(tfp))
	{
	    if (!fgets(buf, 512, tfp))
		break;
	    if (sscanf(buf + 15, "%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d",
		       &consv[0], &consv[1], &consv[2], &consv[3], &consv[4],
		       &consv[5], &consv[6], &consv[7], &consv[8], &consv[9],
		       &consv[10], &consv[11], &consv[12], &consv[13],
		       &consv[14], &consv[15], &consv[16], &consv[17],
		       &consv[18], &consv[19]) != 20)
	    {
		fail("Bad consensus records in dso file!");
	    }
	    else
	    {
		for (j = 0; j < 20; j++)
		    db[nblocks].profile[i][j] = consv[j] / 1000.0;
	    }

	    if (buf[5] == '-' || buf[5] == 'X')
	    {
		db[nblocks].seq[i] = 22;
		db[nblocks].disord[i] = '?';
	    }
	    else
	    {
		db[nblocks].seq[i] = j = aanum(buf[5]);
		db[nblocks].disord[i] = buf[7];
		if (j < 20)
		    db[nblocks].aacomp[j]++;
	    }
	    i++;
	}
	fclose(tfp);
	for (j=0; j<20; j++)
	    db[nblocks].aacomp[j] /= nres;
	restot += nres;
	nblocks++;
    }

    fclose(ifp);

    sampord = malloc(restot * sizeof(struct samp));

    if (sampord == NULL)
	fail("read_data: malloc sampord failed!");
    for (nord = ndis = k = i = 0; i < nblocks; i++)
	for (j = 0; j < db[i].length; j++)
	    if (db[i].disord[j] != '?')
	    {
		if (db[i].disord[j] == '.')
		    nord++;
		else
		    ndis++;

//		printf("%d %d %c\n", i, j, db[i].disord[j]);
		
		sampord[k].testflg = 1;
		sampord[k].en = i;
		sampord[k].wp = j;
		k++;
	    }

    restot = k;

    printf("\n%d blocks read from training set.\n\n", nblocks);
    printf("\n%d residues in training set:\n", restot);
    printf("\n%d Non-disordered residues:\n", nord);
    printf("\n%d Disordered residues:\n", ndis);
}

main(argc, argv)
     int             argc;
     char          **argv;
{
    int             i, niters;

    /* malloc_debug(3); */
    puts("Neural Network Protein Disorder Predictor\n\n");
    if (argc < 3)
	fail("usage : disord_wt weight-file number-of-epochs {train.lst}");
    if (argc > 3)
      read_dat(argv[3]);
    else
      read_dat("train.lst");
    niters = atoi(argv[2]);
    randomize();
    init();
    load_wts(wtfnm = argv[1]);

    testpred();
}
