# Enhanced SMILES Grammar - Overview and Description

### Overview of Changes and Additions to the Grammar

The original SMILES grammar primarily handled organic atoms and simple bonds. The new grammar now supports organometallic complexes and allows representation of a wide variety of chemical environments, making it much more versatile compared to the original version.

### Detailed Descriptions

#### 1. Base Structure Definitions

```bnf
smiles -> chain
chain -> branched_atom
chain -> chain branched_atom
chain -> chain bond branched_atom
chain -> chain bond chain
```

- **Description**: Defines the basic structure of a SMILES string as **chains** built from **branched\_atoms** connected by **bonds**. Chains are the building blocks that can grow by adding atoms or bonds.

#### 2. Branched Atom Definitions

```bnf
branched_atom -> atom
branched_atom -> atom RB
branched_atom -> atom BB
branched_atom -> atom RB BB
branched_atom -> metal_complex
```

- **Description**: A **branched\_atom** can be a standalone atom, or it may include **ring bonds (RB)** and **branches (BB)**, or even a complex metal atom with ligands.

#### 3. Atoms

```bnf
atom -> bracket_atom
atom -> aliphatic_organic
atom -> aromatic_organic
atom -> sulfur_aromatic
atom -> bracketed_atom_symbol
```

- **Description**: Atoms can take different forms: organic atoms, metal atoms, aromatic systems.
- **NOTE**: The inclusion of **bracketed\_atom\_symbol** (e.g., [Sc]) and **sulfur\_aromatic** ensures correct differentiation of certain atoms and avoids ambiguity. Specifically, this handles the distinction between the case when Sc stands for scandium from the case where it stands for an aliphatic sulfur bonded to an aromatic carbon.

#### 4. Handling Transition Metals with Different States

```bnf
bracket_atom -> '[' metal_symbol ']'
bracket_atom -> '[' metal_symbol hcount ']'
bracket_atom -> '[' metal_symbol charge ']'
...
```

- **Description**: Defines transition metal atoms within brackets, supporting different hydrogen counts, charges, or ring bonds. This enables representation of different oxidation states and coordination environments.

#### 5. Atoms Enclosed in Brackets

```bnf
bracketed_atom_symbol -> '[Sc]'
bracketed_atom_symbol -> '[Sc+]'
bracketed_atom_symbol -> '[Sc+ DIGIT]'
```

- **Description**: **bracketed\_atom\_symbol** helps distinguish specific atoms that are always bracketed, e.g., scandium, which would otherwise be confused with the aromatic combination of sulfur and carbon.

#### 6. Handling Complex Metal-Ligand Coordination Structures

```bnf
metal_complex -> '[' metal_symbol complex_ligands ']'
...
```

- **Description**: Represents **metal complexes** with multiple ligands. These rules account for hydrogens, charges, and different types of ligands, thus enabling representation of complex coordination compounds.

#### 7. Distinguishing Sulfur Aromatic from Scandium

```bnf
sulfur_aromatic -> 'S' 'c'
```

- **Description**: Added to explicitly distinguish sulfur followed by aromatic carbon ("Sc") from the metal scandium, which is enclosed in brackets.

#### 8. Metal Symbols

```bnf
metal_symbol -> 'Cd' | 'Os' | 'Ti' | ... | 'Tl'
```

- **Description**: Extended list of **metal symbols** that can be used in brackets to include most transition metals and several other common elements. This extension is critical for organometallic and coordination chemistry.

#### 9. Organic and Inorganic Atoms

```bnf
aliphatic_organic -> 'B' | 'C' | 'F' | ...
aromatic_organic -> 'b' | 'c' | 'n' | ...
```

- **Description**: Expanded the definitions for **organic atoms**, both aliphatic and aromatic, to include additional elements like **Ge**, **Ga**, and **Te**. These additions make the grammar more comprehensive for organic chemistry.

#### 10. Isotopes, Chirality, and Charges

```bnf
BAI -> isotope symbol BAC
BAC -> chiral BAH
BAH -> hcount BACH
BACH -> charge
```

- **Description**: These rules are used to define isotopes, chirality (`@` or `@@`), hydrogens, and charges. This allows for detailed stereochemistry and isotope information, which is essential for accurate representation in SMILES.

#### 11. Bonds and Ring Bonds

```bnf
bond -> '-' | '=' | '#' | '/' | '\\' | ':'
ringbond -> DIGIT | bond DIGIT | '%' DIGIT DIGIT | bond '%' DIGIT DIGIT
```

- **New Addition**: Added `:` to the **bond** definition to represent aromatic bonds, commonly found in coordination compounds.
- **Extended Ring Bond**: Allowed multi-character ring bonds (`%XX`), enabling more complex ring structures.

#### 12. Branch and Ring Handling

```bnf
RB -> RB ringbond | ringbond
BB -> BB branch | branch
branch -> '(' chain ')'
...
```

- **Description**: These rules provide definitions for **ring bonds (RB)** and **branches (BB)**, which are key to representing cyclic structures and branching within molecules.

### Summary of Improvements

- **Expanded Atom Definitions**: Included more elements and atoms in different configurations.
- **Bracket Handling for Metals**: Enhanced handling of metal atoms with various bonding states, charges, and hydrogen counts.
- **Metal-Ligand Complexes**: Added capabilities to represent transition metals with different ligands.
- **Explicit Symbol Handling**: Differentiated between "Sc" as sulfur-carbon and "[Sc]" as scandium.
- **Extended Bonds**: Included aromatic bonds (`:`) and multi-character ring indices.

This enhanced SMILES grammar now provides a significantly more comprehensive representation of chemical structures, suitable for a wide range of molecules, including coordination compounds, organometallics, and more complex organic structures. It is also structured to make future modifications straightforward, thanks to clearly defined components and flexible rules.
